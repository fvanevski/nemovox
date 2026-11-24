"""Bounded private diagnostic: distinguish ASR/CTC loss and test source-first MT.

Not a subtitle workflow, not an automatic acceptance test. No production edits,
historical output injection, model downloads, or unmanaged GPU service changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
    path = root / relative
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
        "generated_lexical_units": len(generated),
        "reconstructed_lexical_units": len(reconstructed),
        "sequence_differences": [
            {"op": op, "generated": list(generated[a:b]),
             "reconstructed": list(reconstructed[c:d])}
            for op, a, b, c, d in diff.get_opcodes() if op != "equal"
        ],
    }


def inspect_prior(root, manifest, output):
    for job, language in (("fr-question", "fr"), ("zh-gap", "zh")):
        recording = verified_prior(root, manifest, f"{job}/asr/recording.contract.json")
        canonical = verified_prior(root, manifest, f"{job}/canonical-source.contract.json")
        quality = verified_prior(root, manifest, f"{job}/source-quality.json")
        chunks = []
        prefix = f"{job}/asr/chunks/"
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
        doc = verified_prior(root, manifest, f"{job}/translated.contract.json")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-root", required=True, type=Path)
    parser.add_argument("--prior-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--fr-media", type=Path)
    parser.add_argument("--zh-media", type=Path)
    parser.add_argument("--model-resources")
    args = parser.parse_args()
    manifest = load(args.prior_manifest)
    root = args.prior_root.resolve()
    out = args.output_dir.resolve()
    if out == root or root in out.parents:
        raise ValueError("Keep new diagnostics outside the prior job root")
    out.mkdir(parents=True, exist_ok=False)
    inspect_prior(root, manifest, out)
    if args.inspect_only:
        return
    if not args.fr_media or not args.zh_media:
        parser.error("Live probe requires both original media paths")
    inputs = {"fr": args.fr_media, "zh": args.zh_media}
    media = {lang: probe_media(path, audio_stream_index=0) for lang, path in inputs.items()}
    files = {}
    for name, lang, start, duration in REGIONS:
        if start + duration > media[lang].duration_s:
            raise ValueError(f"Region exceeds media: {name}")
        wav = out / f"{name}.wav"
        extract_wav_segment(media[lang].path, start_s=start, duration_s=duration,
                            out_wav=wav, audio_stream_index=media[lang].selected_audio_stream)
        actual, digest = validate_wav_file(wav)
        if abs(actual-duration) > .1:
            raise ValueError(f"Unexpected extracted duration: {name}")
        files[name] = {"wav": wav, "sha256": digest, "duration": actual,
                       "start": start, "language": lang}
    registry = load_model_resources(args.model_resources)
    asr = registry.resolve("asr")
    reviewer = registry.resolve("translation_review")
    lifecycle = ModelLifecycle()
    save(out / "inputs.json", {
        "prior_manifest_sha256": sha256(args.prior_manifest),
        "media": {lang: {"path": str(path), "sha256": sha256(path)}
                  for lang, path in inputs.items()},
        "regions": {name: {**item, "wav": str(item["wav"])} for name, item in files.items()},
        "asr_resource": asr.fingerprint(), "review_resource": reviewer.fingerprint(),
        "asr_model": asr.model, "review_model": reviewer.model,
    })
    client = NemotronVLLMClient(base_url=asr.endpoint, model=asr.model,
                              api_key=asr.credential(), timeout_s=900.)
    # One managed ASR acquisition for all short paired mode comparisons.
    with managed_asr_client(lifecycle, asr, client) as live:
        for name, lang, start, _ in REGIONS:
            modes = MODES + (("translate",) if name in ("fr-omission", "zh-gap-second") else ())
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
    # Separate text lease after ASR releases; no simultaneous model loading.
    with managed_text_client(lifecycle, reviewer, wire_protocol="ninfer_messages") as text:
        decisions = text_probe(text, root, manifest, out)
    save(out / "completed.json", {"asr_requests": 14, "text_requests": 8,
                                  "blind_selected_origins": decisions,
                                  "quality_accepted": False})


if __name__ == "__main__":
    main()
