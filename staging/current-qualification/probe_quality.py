"""Bounded private diagnostic: distinguish ASR/CTC loss and test source-first MT.

Not a subtitle workflow, not an automatic acceptance test. No production edits,
historical output injection, model downloads, or unmanaged GPU service changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from difflib import SequenceMatcher
from pathlib import Path

from nvidia_transcribe.client import InferenceRequestConfig, NemotronVLLMClient
from nvidia_transcribe.contract import build_transcript_contract
from nvidia_transcribe.media import extract_wav_segment, probe_media, validate_wav_file
from nvidia_transcribe.model_resources import load_model_resources
from nvidia_transcribe.resource_lifecycle import (
    ModelLifecycle, managed_asr_client, managed_text_client,
)
from nvidia_transcribe.source_quality import _lex
from nvidia_transcribe.asr_integrity import audit_asr_generation

# Original-media seconds; these are diagnostic regions, not expected text.
REGIONS = (
    ("fr-omission", "fr", 789.5, 8.0),
    ("zh-gap-first", "zh", 238.0, 12.0),
    ("zh-gap-second", "zh", 247.0, 13.0),
    ("fr-control", "fr", 248.0, 9.0),
)
MODES = ("speaker", "verbatim", "multispeaker_sot")
# Compare audio geometry at a fixed ASR preset. Nominal cores meet at the
# midpoint; inference contexts overlap on both sides of that boundary.
TARGETS = (
    ("fr-omission", "fr", "fr-question", 770.0, 789.5, 797.5),
    ("zh-gap", "zh", "zh-gap", 195.0, 238.0, 260.0),
)
CONTEXT_PAD_S = 10.0
HALF_OVERLAP_S = 6.0

def context_regions(media):
    for target, lang, _, _, a, b in TARGETS:
        lo, hi = max(0., a-CONTEXT_PAD_S), min(media[lang].duration_s, b+CONTEXT_PAD_S)
        midpoint = (a+b)/2
        for variant, start, end, core in (
            ("expanded", lo, hi, (a,b)),
            ("shift-left", lo, min(hi, midpoint+HALF_OVERLAP_S), (a,midpoint)),
            ("shift-right", max(lo, midpoint-HALF_OVERLAP_S), hi, (midpoint,b)),
        ):
            if not 0 <= start < end <= media[lang].duration_s:
                raise ValueError("Invalid alternative context")
            yield f"{target}-{variant}", lang, start, end-start, {
                "target": target, "variant": variant, "target_bounds": [a,b],
                "nominal_core": list(core),
                "nominal_boundary": midpoint if variant != "expanded" else None,
            }
TEXT_CASES = (
    ("fr-control", "fr", "utt-0002"),
    ("zh-gap", "zh", "utt-0014"),
    ("fr-control", "fr", "utt-0009"),
    ("zh-gap", "zh", "utt-0010"),
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verified_prior(root, manifest, relative):
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("Manifest member escaped prior job root")
    record = manifest["files"].get(relative)
    if not record or not path.is_file() or path.stat().st_size != record["size"]:
        raise ValueError(f"Prior artifact missing or wrong size: {relative}")
    if sha256(path) != record["sha256"]:
        raise ValueError(f"Prior artifact digest mismatch: {relative}")
    return load(path)


def extract_raw(doc):
    if isinstance(doc, dict) and isinstance(doc.get("choices"), list):
        return doc
    if isinstance(doc, dict) and isinstance(doc.get("raw_response"), dict):
        return extract_raw(doc["raw_response"])
    raise ValueError("Unrecognized saved raw ASR response")


def describe(raw, contract):
    raw_text = (raw.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    generated = _lex(contract.get("source_text") or raw_text)
    rows = contract.get("utterances") or []
    reconstructed = tuple(token for row in rows for token in _lex(row.get("source_text")))
    # Lexical comparison is a diagnostic, NOT a correctness gate. Speaker
    # order, segmentation and punctuation can legitimately differ.
    diff = SequenceMatcher(None, generated, reconstructed, autojunk=False)
    return {
        "integrity": audit_asr_generation(raw),
        "raw_text": raw_text,
        "generated_source_text": contract.get("source_text"),
        "timestamp_provenance": contract.get("timestamp_provenance"),
        "diarization_intervals": contract.get("diarization_intervals"),
        "rows": rows,
        "raw_ctc_payload": raw.get("ctc_timestamps"),
        "generated_lexical_units": len(generated),
        "reconstructed_lexical_units": len(reconstructed),
        "sequence_differences": [
            {"op": op, "generated": list(generated[a:b]),
             "reconstructed": list(reconstructed[c:d])}
            for op, a, b, c, d in diff.get_opcodes() if op != "equal"
        ],
    }


def job_paths(manifest, job):
    """Resolve only manifest-bound CLI or Studio job roots, never newest-by-mtime."""
    files = manifest["files"]
    direct = f"{job}/asr/recording.contract.json"
    if direct in files:
        return f"{job}/asr", job
    records = [name for name in files if name.startswith(f"{job}/jobs/")
               and name.endswith("/asr/recording.contract.json")]
    if len(records) != 1:
        raise ValueError(f"Expected one manifest-bound Studio recording for {job}")
    base = records[0].removesuffix("/asr/recording.contract.json")
    return base + "/asr", base + "/output"


def inspect_prior(root, manifest, output):
    for job, language in (("fr-question", "fr"), ("zh-gap", "zh")):
        asr_dir, result_dir = job_paths(manifest, job)
        recording = verified_prior(root, manifest, f"{asr_dir}/recording.contract.json")
        canonical = verified_prior(root, manifest, f"{result_dir}/canonical-source.contract.json")
        quality = verified_prior(root, manifest, f"{result_dir}/source-quality.json")
        chunks = []
        prefix = f"{asr_dir}/chunks/"
        for name in sorted(manifest["files"]):
            if name.startswith(prefix) and name.endswith(".raw.json"):
                raw = extract_raw(verified_prior(root, manifest, name))
                text = (raw.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                # Deliberately use crop-local times here; the recording's
                # actual chunk metadata is retained alongside the projection.
                rebuilt = build_transcript_contract(
                    raw_text=text, source_language=language, ctc_payload=raw.get("ctc_timestamps"),
                    source_offset_s=0.0,
                )
                chunks.append({"file": name, "time_basis": "chunk_local_projection",
                               **describe(raw, rebuilt)})
        probes = []
        for index, obs in enumerate(quality.get("acoustic_observations") or []):
            raw = extract_raw(obs["raw_response"])
            text = (raw.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            rebuilt = build_transcript_contract(
                raw_text=text, source_language=language, ctc_payload=raw.get("ctc_timestamps"),
                source_offset_s=obs["crop"][0],
            )
            probes.append({"index": index, "crop": obs["crop"],
                           "time_basis": "prior_clip_relative",
                           "admitted_observations": obs["utterances"],
                           **describe(raw, rebuilt)})
        save(output / f"{job}-prior-inspection.json", {
            "original_rows": recording["utterances"],
            "canonical_rows": canonical["utterances"],
            "chunk_metadata": recording["processing"]["chunks"],
            "chunks": chunks, "acoustic_probes": probes,
            "note": "Reprojection is diagnostic, not a new acceptance run.",
        })


def structured(client, system, payload, schema, budget=2048):
    start = time.perf_counter()
    reply = client.complete(
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        max_tokens=4096, temperature=0.0,
        response_format={"type": "json_schema", "schema": schema},
        thinking_budget=budget, reasoning_effort="xhigh",
        strict_reasoning_effort=True,
    )
    choice = (reply.get("choices") or [{}])[0]
    if choice.get("finish_reason") != "stop":
        raise RuntimeError("Incomplete bounded text diagnostic")
    parsed = json.loads(choice["message"]["content"])
    return {"parsed": parsed, "response": reply,
            "elapsed_s": round(time.perf_counter() - start, 3)}


def string_schema(*fields):
    return {"type": "object", "properties": {f: {"type": "string"} for f in fields},
            "required": list(fields), "additionalProperties": False}


def text_probe(client, root, manifest, output):
    results = []
    for number, (job, language, uid) in enumerate(TEXT_CASES):
        _, result_dir = job_paths(manifest, job)
        doc = verified_prior(root, manifest, f"{result_dir}/translated.contract.json")
        rows = doc["utterances"]
        indices = [i for i, row in enumerate(rows)
                   if (row.get("utterance_id") or row.get("id")) == uid]
        if len(indices) != 1:
            raise ValueError(f"Nonunique text target: {job}/{uid}")
        i = indices[0]
        row = rows[i]
        source = {"source_language": language, "target": row["source_text"],
                  "context_before": [r["source_text"] for r in rows[max(0, i-2):i]],
                  "context_after": [r["source_text"] for r in rows[i+1:i+3]]}
        first = structured(
            client,
            "Interpret the source dialogue before seeing any candidate English. "
            "Identify the actual action, idioms, polarity, agents and chronology. "
            "Give a literal meaning gloss, then fluent concise English subtitle dialogue. "
            "Where source ASR seems corrupted, distinguish uncertain words from supported meaning; "
            "do not invent named entities or a missing event. Do not assume a literal cognate is correct.",
            source, string_schema("meaning_gloss", "uncertainty", "english"),
        )
        candidate = first["parsed"]["english"]
        if not isinstance(candidate, str) or not candidate.strip():
            raise ValueError("Empty source-first translation")
        choices = [
            ("primary_mt", row["machine_translated_text"]),
            ("previous_review", row.get("translation_reviewed_text") or row["translated_text"]),
            ("source_first", candidate),
        ]
        # Blind candidate labels and rotate order deterministically. The
        # earlier reviewers' rationales and decisions are NOT supplied.
        choices = choices[number % 3:] + choices[:number % 3]
        blind = {chr(65+j): value for j, (_, value) in enumerate(choices)}
        schema = string_schema("meaning_gloss", "comparison", "selected", "uncertainty")
        schema["properties"]["selected"]["enum"] = ["A", "B", "C", "none"]
        second = structured(
            client,
            "Evaluate each anonymous English candidate against the original source, independently "
            "derive the source meaning and explain material action, idiom, polarity and chronology "
            "differences before selecting. Check whether the English words actually express the gloss. "
            "If no option is faithful, select none. No third rewrite or unsupported reconstruction.",
            {**source, "candidates": blind}, schema,
        )
        label = second["parsed"]["selected"]
        if label not in ("A", "B", "C", "none"):
            raise ValueError("Unknown blind choice")
        result = {
            "job": job, "id": uid, "source": source,
            "first": first, "blind_candidates": blind,
            "blind_labels": {chr(65+j): kind for j, (kind, _) in enumerate(choices)},
            "second": second, "previous_final": row["translated_text"],
            "selected_origin": "none" if label == "none" else choices[ord(label)-65][0],
        }
        save(output / f"text-{number:02d}.json", result)
        results.append(result["selected_origin"])
    return results


def snapshot_frames(media, output):
    """CPU-only private frames at scene-relevant requested original times."""
    frames = []
    for target, lang, _, _, a, b in TARGETS:
        for n, time_s in enumerate(sorted(set(
                max(0., min(media[lang].duration_s-.05, t))
                for t in (a-2, a+.25, (a+b)/2, b-.25, b+2)))):
            frame_id = f"{target}-{n:02d}"
            path = output / f"{frame_id}.png"
            command = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "info",
                       "-ss", f"{time_s:.3f}", "-i", str(media[lang].path),
                       "-map", "0:V:0", "-frames:v", "1",
                       "-vf", "showinfo,scale='min(960,iw)':-2", "-y", str(path)]
            row = {"frame_id": frame_id, "target": target, "requested_seek_s": time_s,
                   "time_basis": "original media seek, nearest decoded frame; see showinfo",
                   "path": str(path)}
            try:
                proc = subprocess.run(command, capture_output=True, text=True, timeout=60)
                (output / f"{frame_id}.ffmpeg.log").write_text(proc.stderr, encoding="utf-8")
                row["exit"] = proc.returncode
                if proc.returncode == 0 and path.is_file() and path.stat().st_size:
                    row["sha256"] = sha256(path)
                    row["status"] = "captured"
                else:
                    row["status"] = "unavailable"
            except (OSError, subprocess.TimeoutExpired) as exc:
                row.update(status="unavailable", error=type(exc).__name__)
            frames.append(row)
    return frames


def nearby_rows(rows, a, b, *, offset=0.):
    result = []
    for i, row in enumerate(rows):
        start, end = row.get("source_start"), row.get("source_end")
        if start is None or end is None:
            # Retain untimed hypotheses with their original list order.
            bounds = None
        else:
            bounds = [float(start)+offset, float(end)+offset]
            if bounds[1] < a-12 or bounds[0] > b+12:
                continue
        result.append({
            "id": row.get("observation_id") or row.get("id") or row.get("utterance_id") or str(i),
            "row_order": i, "source_text": row.get("source_text"),
            "bounds": bounds,
            "voice": row.get("local_voice") or row.get("speaker_id"),
        })
    return result


def prior_context(root, manifest, target):
    name, lang, job, offset, a, b = target
    asr_dir, result_dir = job_paths(manifest, job)
    recording = verified_prior(root, manifest, f"{asr_dir}/recording.contract.json")
    quality = verified_prior(root, manifest, f"{result_dir}/source-quality.json")
    observations = []
    for n, obs in enumerate(quality.get("acoustic_observations", [])):
        ca, cb = [float(t)+offset for t in obs["crop"]]
        if cb < a or ca > b:
            continue
        observations.append({
            "id": f"prior-probe-{n:03d}", "crop": [ca,cb],
            "audio_sha256": obs["audio_sha256"], "integrity": obs.get("integrity"),
            "rows": nearby_rows(obs["utterances"], a, b, offset=offset),
        })
    return {
        "case": name, "source_language": lang, "target_bounds": [a,b],
        "time_basis": "original media seconds",
        "surrounding_source": nearby_rows(recording["utterances"], a,b,offset=offset),
        "acoustic_hypotheses": observations,
        "limitations": "Same checkpoint; voice labels are local hypotheses. Null bounds are untimed, not silence.",
    }


def assessment_schema():
    schema = string_schema("decision", "supported_meaning", "candidate_source",
                           "timing_assessment", "evidence_and_counterevidence",
                           "remaining_uncertainty", "next_evidence")
    schema["properties"]["decision"]["enum"] = [
        "meaning_resolved", "meaning_resolved_timing_open",
        "materially_ambiguous", "no_supported_speech",
    ]
    schema["properties"]["next_evidence"]["enum"] = [
        "none", "expanded_context", "shifted_boundary", "independent_asr",
        "visual_scene", "alignment", "human_listening",
    ]
    return schema


ASSESS_SYSTEM = (
    "Resolve material subtitle ambiguity from the supplied evidence. Reason openly about "
    "competing source meanings, actual occurrence, chronology, local speaker turns and timing. "
    "Use surrounding dialogue to distinguish plausible misrecognitions, idioms and referents. "
    "A contextual inference can resolve meaning without additional audio if positive evidence "
    "discriminates alternatives, but it cannot establish unheard speech, exact lexical truth, "
    "speaker identity or missing timestamps. Give the least speculative source-language "
    "rendering consistent with the supported meaning; leave it empty if unsupported. "
    "Cite actual observation and row IDs and counterevidence; distinguish temporal bounds "
    "from wording, visual observation from interpretation, and silence from missing data. "
    "Do not treat multiple crops of one model as independent votes. Identify the one next "
    "informative evidence type, or none if the remaining uncertainty is immaterial. "
    "If visually supplied, do not infer exact words from mouth motion, and distinguish "
    "visible character action, objects, on-screen text and reaction from spoken content."
)


def collect_assessment_inputs(collected, inputs, target, independent, visual):
    name, lang, _, _, a, b = target
    observations = []
    for region, item in inputs["regions"].items():
        if item["language"] != lang or item.get("target") not in (None, name):
            continue
        if item["start"] >= b or item["start"]+item["duration"] <= a:
            continue
        wav = Path(item["wav"])
        if not wav.is_file() or sha256(wav) != item["sha256"]:
            raise ValueError(f"Collected audio identity mismatch: {region}")
        for path in sorted(collected.glob(region+"-*.json")):
            # Exact region prefix plus a known mode, not a neighboring
            # expanded-region filename.
            mode = path.stem[len(region)+1:]
            if mode not in (*MODES, "translate"):
                continue
            record = load(path)
            if record["audio_sha256"] != item["sha256"] or record["mode"] != mode:
                raise ValueError(f"ASR receipt mismatch: {path.name}")
            contract = record["contract"]
            observations.append({
                "id": path.stem, "mode": mode, "audio_sha256": item["sha256"],
                "crop": [item["start"], item["start"]+item["duration"]],
                "nominal_core": item.get("nominal_core"),
                "integrity": audit_asr_generation(record["raw_response"]),
                "rows": nearby_rows(contract["utterances"], a,b),
                "direct_english_diagnostic": contract.get("english_translation") if mode=="translate" else None,
            })
    independent_rows = []
    if independent:
        for region, result in independent["results"].items():
            item = inputs["regions"].get(region)
            if not item or result["audio_sha256"] != item["sha256"]:
                raise ValueError("Independent acoustic input mismatch")
            if item["language"] == lang and item["start"] < b and item["start"]+item["duration"] > a:
                independent_rows.append({"id": f"independent-{region}",
                                         "rows": result["rows"]})
    frames = {f["frame_id"]: f for f in inputs.get("frames", []) if f["status"]=="captured"}
    descriptions = []
    if visual:
        for desc in visual["frames"]:
            frame = frames.get(desc["frame_id"])
            if not frame or frame["sha256"] != desc["sha256"] or sha256(frame["path"]) != frame["sha256"]:
                raise ValueError("Visual evidence frame mismatch")
            if frame["target"] == name:
                descriptions.append({**desc, "requested_seek_s": frame["requested_seek_s"]})
    return {
        "fresh_acoustic_observations": observations,
        "independent_asr": {"identity": independent.get("checkpoint"),
                            "rows": independent_rows} if independent else None,
        "visual_observations": {"identity": visual["observer_identity"],
                                "frames": descriptions} if visual else None,
    }


def run_assessments(client, root, manifest, collected, output, independent, visual):
    inputs = load(collected / "inputs.json")
    if inputs["prior_manifest_sha256"] != sha256(manifest["_path"]):
        raise ValueError("Collected evidence belongs to another prior manifest")
    for target in TARGETS:
        name = target[0]
        prior = prior_context(root, manifest, target)
        # Evidence-isolated counterfactual: this call receives only evidence
        # available BEFORE acoustic escalation; no fresh or visual observations.
        before = structured(client, ASSESS_SYSTEM, prior, assessment_schema())
        save(output / f"{name}-context-only.json", {"input": prior, "result": before})
        fresh = collect_assessment_inputs(collected, inputs, target, independent, visual)
        fused_input = {**prior, **fresh}
        # A separate call, not conditioned on the first verdict/rationale.
        after = structured(client, ASSESS_SYSTEM, fused_input, assessment_schema())
        save(output / f"{name}-integrated.json", {"input": fused_input, "result": after})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-root", required=True, type=Path)
    parser.add_argument("--prior-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--inspect-only", action="store_true")
    phase.add_argument("--collect-only", action="store_true")
    phase.add_argument("--assess-collected", type=Path)
    parser.add_argument("--independent-asr", type=Path)
    parser.add_argument("--visual-observations", type=Path)
    parser.add_argument("--fr-media", type=Path)
    parser.add_argument("--zh-media", type=Path)
    parser.add_argument("--model-resources")
    args = parser.parse_args()
    manifest = load(args.prior_manifest)
    manifest["_path"] = str(args.prior_manifest.resolve())
    root = args.prior_root.resolve()
    out = args.output_dir.resolve()
    if out == root or root in out.parents:
        raise ValueError("Keep new diagnostics outside the prior job root")
    if Path(manifest["host_local_root"]).resolve() != root:
        raise ValueError("Prior root does not match verified manifest")
    out.mkdir(parents=True, exist_ok=False)
    if args.assess_collected:
        collected = args.assess_collected.resolve()
        independent = load(args.independent_asr) if args.independent_asr else None
        visual = load(args.visual_observations) if args.visual_observations else None
        registry = load_model_resources(args.model_resources)
        reviewer = registry.resolve("translation_review")
        with managed_text_client(ModelLifecycle(), reviewer, wire_protocol="ninfer_messages") as text:
            run_assessments(text, root, manifest, collected, out, independent, visual)
            decisions = text_probe(text, root, manifest, out)
        save(out / "completed.json", {"text_requests": 12, "blind_selected_origins": decisions,
                                      "quality_accepted": False})
        return
    inspect_prior(root, manifest, out)
    if args.inspect_only:
        return
    if not args.fr_media or not args.zh_media:
        parser.error("Live probe requires both original media paths")
    inputs = {"fr": args.fr_media, "zh": args.zh_media}
    media = {lang: probe_media(path, audio_stream_index=0) for lang, path in inputs.items()}
    files = {}
    geometry = {name: {"variant": "base"} for name, *_ in REGIONS}
    requests = list(REGIONS)
    for name, lang, start, duration, metadata in context_regions(media):
        geometry[name] = metadata
        requests.append((name,lang,start,duration))
    for name, lang, start, duration in requests:
        if start + duration > media[lang].duration_s:
            raise ValueError(f"Region exceeds media: {name}")
        wav = out / f"{name}.wav"
        extract_wav_segment(media[lang].path, start_s=start, duration_s=duration,
                            out_wav=wav, audio_stream_index=media[lang].selected_audio_stream)
        actual, digest = validate_wav_file(wav)
        if abs(actual-duration) > .1:
            raise ValueError(f"Unexpected extracted duration: {name}")
        files[name] = {"wav": wav, "sha256": digest, "duration": actual,
                       "start": start, "language": lang, **geometry[name]}
    frames = snapshot_frames(media, out)
    registry = load_model_resources(args.model_resources)
    asr = registry.resolve("asr")
    reviewer = registry.resolve("translation_review")
    lifecycle = ModelLifecycle()
    save(out / "inputs.json", {
        "prior_manifest_sha256": sha256(args.prior_manifest),
        "media": {lang: {"path": str(path), "sha256": sha256(path)}
                  for lang, path in inputs.items()},
        "regions": {name: {**item, "wav": str(item["wav"])} for name, item in files.items()},
        "frames": frames,
        "asr_resource": asr.fingerprint(), "review_resource": reviewer.fingerprint(),
        "asr_model": asr.model, "review_model": reviewer.model,
    })
    client = NemotronVLLMClient(base_url=asr.endpoint, model=asr.model,
                              api_key=asr.credential(), timeout_s=900.)
    # One managed ASR acquisition for all short paired mode comparisons.
    with managed_asr_client(lifecycle, asr, client) as live:
        for name, lang, start, _ in requests:
            modes = (MODES + (("translate",) if name in ("fr-omission", "zh-gap-second") else ())
                     if geometry[name]["variant"] == "base" else ("speaker",))
            for mode in modes:
                cfg = InferenceRequestConfig(
                    mode=mode, source_language=lang, source_offset_s=start,
                    target_language="English", capture_ctc_timestamps=mode != "translate",
                    apply_loop_guardrail=False,
                )
                result = live.transcribe(files[name]["wav"], cfg)
                raw, contract = result["raw_response"], result["contract"]
                save(out / f"{name}-{mode}.json", {
                    "mode": mode, "language": lang, "audio_sha256": files[name]["sha256"],
                    "source_offset_s": start, "latency_s": result["latency_s"],
                    "raw_response": raw, "contract": contract,
                    "diagnostic": (None if mode == "translate" else describe(raw, contract)),
                    "note": ("direct translation is diagnostic only; not timed source evidence"
                             if mode == "translate" else "same-model alternative prompt"),
                })
    save(out / "collection-complete.json", {"asr_requests": 20,
                                            "quality_accepted": False})
    if args.collect_only:
        return
    # Default combined mode remains useful when no external observations are available.
    with managed_text_client(lifecycle, reviewer, wire_protocol="ninfer_messages") as text:
        run_assessments(text, root, manifest, out, out, None, None)
        decisions = text_probe(text, root, manifest, out)
    save(out / "completed.json", {"asr_requests": 20, "text_requests": 12,
                                  "blind_selected_origins": decisions,
                                  "quality_accepted": False})


if __name__ == "__main__":
    main()
