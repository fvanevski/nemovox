"""Bounded matched acoustic geometry evaluation using the current recording pipeline.

The two policy arms consume identical PCM region bytes. Two distinct local
crops supply positive same-model reference events, not acoustic ground truth.
All times in the private comparison are mapped back to original media time.
No production defaults or model lifecycle policy are changed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import tarfile
import time

from nvidia_transcribe.client import NemotronVLLMClient
from nvidia_transcribe.media import extract_wav_segment, probe_media, validate_wav_file
from nvidia_transcribe.model_resources import load_model_resources
from nvidia_transcribe.recording import RecordingPipelineConfig, process_media
from nvidia_transcribe.resource_lifecycle import ModelLifecycle, managed_asr_client
from nvidia_transcribe.source_quality import (
    _bounds, _contains, _lex, _observation_groups, _same_event_bounds,
    corroborated_events,
)

DIAGNOSTIC_SHA = "307c0a32bd4ae32008be0141d5ac12e066f96a728bfcb179c0b343bf2031fd7b"
CHECKPOINT = "/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"
TOLERANCE_S = .85
MAX_REQUESTS = 72
MAX_INFERENCE_AUDIO_S = 2000.
# Preregistered evaluation intervals, NOT production settings or acoustic labels.
# All four policy comparisons have identical 110s coverage.
REGIONS = (
    ("compression", (260., 370.), (298., 322.), ((286., 329.), (291., 335.))),
    ("boundary", (570., 680.), (609., 643.), ((599., 650.), (605., 658.))),
    ("late_gap", (1160., 1270.), (1212., 1252.), ((1201., 1259.), (1207., 1265.))),
    ("low_flag_control", (440., 550.), (470., 505.), ((460., 511.), (465., 517.))),
)
POLICIES = {
    "fixed": {"chunk_policy": "fixed", "chunk_seconds": 90., "context_seconds": 10.},
    "pause_aware": {"chunk_policy": "pause_aware", "chunk_seconds": 46., "context_seconds": 4.},
}


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_authority(path, expected_sha=DIAGNOSTIC_SHA):
    """Read exact retained diagnostic authority; never extract archive members."""
    if sha_file(path) != expected_sha:
        raise ValueError("Diagnostic archive digest mismatch")
    ledgers, raws, sources = {}, {}, []
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            if member.size > 32 * 1024 * 1024:
                raise ValueError("Oversize JSON in diagnostic archive")
            data = archive.extractfile(member).read()
            try:
                doc = json.loads(data)
            except (ValueError, UnicodeError):
                continue
            if not isinstance(doc, dict):
                continue
            digest = hashlib.sha256(data).hexdigest()
            if "asr_observations" in doc and "media_fingerprint_sha256" in doc:
                ledgers[digest] = doc
            if doc.get("choices") and "ctc_timestamps" in doc:
                raws[digest] = doc
            for source in (doc, doc.get("source"), doc.get("media")):
                if isinstance(source, dict) and "fingerprint_sha256" in source:
                    sources.append(source)
    eligible = [d for d in ledgers.values() if len(d["asr_observations"]) == 16
                and len({r["chunk_index"] for r in d["asr_observations"]}) == 16
                and all(r["raw_response_sha256"] in raws for r in d["asr_observations"])]
    if len(eligible) != 1:
        raise ValueError("Expected unique complete 16-response source ledger")
    ledger = eligible[0]
    fingerprint = ledger["media_fingerprint_sha256"]
    matching = [s for s in sources if s["fingerprint_sha256"] == fingerprint
                and "selected_audio_stream" in s and "duration_s" in s]
    identities = {(int(s["selected_audio_stream"]), float(s["duration_s"])) for s in matching}
    if len(identities) != 1:
        raise ValueError("Original media stream/duration metadata missing or conflicting")
    stream, duration = next(iter(identities))
    if ledger["source_language"] != "fr":
        raise ValueError("Geometry intervals are registered for the French diagnostic only")
    models = sorted({str(raws[r["raw_response_sha256"]]["model"])
                     for r in ledger["asr_observations"]
                     if raws[r["raw_response_sha256"]].get("model")})
    if len(models) > 1:
        raise ValueError("Diagnostic contains multiple served model identities")
    return {"fingerprint_sha256": fingerprint, "selected_audio_stream": stream,
            "duration_s": duration, "source_language": "fr", "served_models": models,
            "media_paths": sorted({s["path"] for s in matching if s.get("path")})}


def validate_regions(duration_s):
    for name, region, target, probes in REGIONS:
        a, b = region
        x, y = target
        if not (0 <= a < x < y < b <= duration_s and b-a == 110.):
            raise ValueError(f"Invalid matched region {name}")
        if len(set(probes)) != 2 or any(not a <= c < x < y < d <= b for c, d in probes):
            raise ValueError(f"Invalid independent-context crops for {name}")


def validate_partition(chunks, duration):
    if not chunks or abs(chunks[0]["core_start_s"]) > .001:
        raise ValueError("Missing first core")
    if abs(chunks[-1]["core_end_s"] - duration) > .001:
        raise ValueError("Incomplete final core")
    for i, c in enumerate(chunks):
        if not (c["chunk_index"] == i and
                0 <= c["infer_start_s"] <= c["core_start_s"] < c["core_end_s"]
                <= c["infer_end_s"] <= duration + .001):
            raise ValueError("Invalid effective crop/core bounds")
        if i and abs(c["core_start_s"] - chunks[i-1]["core_end_s"]) > .001:
            raise ValueError("Core partition has gap or overlap")


class BudgetedClient:
    """Bound actual inference requests (including normal retry/subdivision)."""
    def __init__(self, client, journal):
        self.client = client
        self.journal = journal
        self.requests = []
        self.run = None

    def __getattr__(self, key):
        return getattr(self.client, key)

    def transcribe(self, audio_path, config=None):
        duration, digest = validate_wav_file(audio_path)
        if len(self.requests) >= MAX_REQUESTS or sum(r["audio_seconds"] for r in self.requests)+duration > MAX_INFERENCE_AUDIO_S:
            raise RuntimeError("Registered geometry evaluation inference budget exhausted")
        receipt = {"run": self.run, "audio_seconds": duration, "audio_sha256": digest,
                   "source_offset_s": getattr(config, "source_offset_s", None),
                   "status": "started"}
        self.requests.append(receipt)
        write_json(self.journal, self.requests)
        started = time.perf_counter()
        try:
            result = self.client.transcribe(audio_path, config)
            raw = result["raw_response"]
            receipt["status"] = "returned"
            receipt["finish_reason"] = (raw.get("choices") or [{}])[0].get("finish_reason")
            receipt["ctc_present"] = "ctc_timestamps" in raw
            receipt["ctc_error"] = bool((raw.get("ctc_timestamps") or {}).get("error"))
            return result
        except Exception as exc:
            receipt["status"] = "failed"
            receipt["error_type"] = type(exc).__name__
            raise
        finally:
            receipt["wall_s"] = round(time.perf_counter()-started, 4)
            write_json(self.journal, self.requests)


def inspect_run(out, crop, wav_sha):
    contract = out["contract"]
    ledger_path = Path(out["source_evidence_path"])
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    manifest = json.loads(Path(out["manifest_path"]).read_text(encoding="utf-8"))
    if manifest.get("asr_run_status") != "completed":
        raise ValueError("Recording did not complete")
    if manifest["source_evidence"]["sha256"] != sha_file(ledger_path):
        raise ValueError("Source evidence digest mismatch")
    chunks = manifest["chunks"]
    validate_partition(chunks, contract["source"]["duration_s"])
    for c in chunks:
        if (c["generation_integrity"]["status"] != "complete"
                or sha_file(c["raw_response_path"]) != c["raw_response_sha256"]):
            raise ValueError("Unaccepted or unauthenticated leaf response")
    units = ledger["lexical_observations"]
    by_id = {u["id"]: u for u in units}
    if len(by_id) != len(units):
        raise ValueError("Duplicate source-unit identity")
    inventories = ledger["recognized_occurrence_inventories"]
    inv_ids = [uid for inv in inventories for uid in inv["working_source_unit_ids"]]
    if len(inv_ids) != len(set(inv_ids)) or set(inv_ids) != set(by_id):
        raise ValueError("Original working source inventory incomplete")
    spans = [s for inv in inventories for s in inv["locality"]["spans"]]
    span_ids = [uid for s in spans for uid in s["source_unit_ids"]]
    if sorted(span_ids) != sorted(inv_ids):
        raise ValueError("Occurrence span partition incomplete")
    for u in units:
        decision = u["boundary_decision"]
        if decision["source_unit_id"] != u["id"] or not set(decision["related_source_unit_ids"]) <= set(by_id):
            raise ValueError("Invalid source disposition references")
    selected = {u["id"] for u in units if u["boundary_disposition"].startswith("selected_")}
    rows = contract["utterances"]
    rendered_ids = [uid for row in rows for uid in row["source_unit_ids"]]
    if len(rendered_ids) != len(set(rendered_ids)) or set(rendered_ids) != selected:
        raise ValueError("Recording utterances do not cover selected observations")
    disputed = {s["id"] for s in spans if s["locality_status"] == "disputed"}
    all_rows, timed_rows = [], []
    for row in rows:
        item = {**row, "observation_id": row["id"], "local_voice": row["speaker_id"]}
        for key in ("source_start", "source_end"):
            if item.get(key) is not None:
                item[key] = float(item[key]) + crop[0]
        all_rows.append(item)
        bounds = _bounds(item)
        if (bounds and crop[0]-.01 <= bounds[0] < bounds[1] <= crop[1]+.01
                and row.get("untimed_unit_count", 0) == 0
                and not disputed.intersection(row.get("occurrence_span_ids", []))
                and _lex(item["source_text"])):
            timed_rows.append(item)
    partial = sum(0 < len(selected.intersection(s["source_unit_ids"])) < len(s["source_unit_ids"])
                  for s in spans if s["locality_status"] == "provisional")
    metrics = {
        "crop": list(crop), "audio_sha256": wav_sha,
        "chunks": len(chunks), "max_infer_s": max(c["infer_end_s"]-c["infer_start_s"] for c in chunks),
        "chunk_plan": manifest["chunk_plan"],
        "projection_statuses": dict(Counter(inv["projection_status"] for inv in inventories)),
        "recognized_turns": sum(len(inv["recognized_turns"]) for inv in inventories),
        "native_observations": sum(len(inv["native_ctc_observations"]) for inv in inventories),
        "source_units": len(units),
        "dispositions": dict(Counter(u["boundary_disposition"] for u in units)),
        "locality_statuses": dict(Counter(s["locality_status"] for s in spans)),
        "partial_provisional_spans": partial,
        "utterances": len(rows), "strictly_timed_rows": len(timed_rows),
        "wholly_untimed_rows": sum(r["source_start"] is None for r in rows),
        "wall_s": out["total_wall_time_s"],
        "raw_response_sha256": [c["raw_response_sha256"] for c in chunks],
    }
    # IDs need per-run namespace in matching, since the same crop can emit
    # identical observations across policies. Never compare ordinal IDs across crops.
    return metrics, {"crop": list(crop), "audio_sha256": wav_sha,
                     "raw_response_sha256": sha_file(out["manifest_path"]),
                     "utterances": timed_rows, "all_rows": all_rows,
                     "raw_turn_lexemes": [_lex(t["text"]) for inv in inventories
                                          for t in inv["recognized_turns"]]}


def compare_reference(events, arm):
    """Strict event correspondence against fixed same-model positive support.

    Unmatched strict text is not proof of omission. Multiple exact matches
    are not collapsed merely because their lexical forms are equal.
    """
    groups = _observation_groups(arm)
    rows = arm["all_rows"]
    decisions = []
    for event in events:
        key = _lex(event["source_text"])
        target = _bounds(event)
        lexical = [g for g in groups if _lex(g["source_text"]) == key]
        same_time = [g for g in lexical if _same_event_bounds(_bounds(g), target, TOLERANCE_S)]
        d = {"reference_ids": event["supporting_observation_ids"],
             "reference_bounds_s": list(target),
             "raw_turn_contains_exact_lexical_sequence": any(_contains(t, key) for t in arm["raw_turn_lexemes"]),
             "selected_row_contains_exact_lexical_sequence": any(_contains(_lex(r["source_text"]), key) for r in rows),
             "matching_candidates": [{"ids": g["member_ids"], "bounds_s": list(_bounds(g))}
                                     for g in lexical],
             "status": "no_strict_event_match"}
        if len(same_time) == 1:
            g = same_time[0]
            bounds = _bounds(g)
            d.update(status="same_model_timing_support", matched_ids=g["member_ids"],
                     onset_delta_s=round(bounds[0]-target[0], 4),
                     end_delta_s=round(bounds[1]-target[1], 4))
        elif len(same_time) > 1:
            d["status"] = "ambiguous_same_time_matches"
        elif lexical:
            d["status"] = "exact_text_different_timing_unresolved"
        elif any(_bounds(r) is None and _lex(r["source_text"]) == key for r in rows):
            d["status"] = "exact_text_unlocalized"
        decisions.append(d)
    counts = Counter(uid for d in decisions for uid in d.get("matched_ids", []))
    for d in decisions:
        if any(counts[uid] > 1 for uid in d.get("matched_ids", [])):
            d["status"] = "ambiguous_reused_match"
            for k in ("matched_ids", "onset_delta_s", "end_delta_s"):
                d.pop(k, None)
    return decisions


def comparison_metrics(decisions):
    supported = [d for d in decisions if d["status"] == "same_model_timing_support"]
    return {"statuses": dict(Counter(d["status"] for d in decisions)),
            "reference_events": len(decisions),
            "raw_exact_sequence_support": sum(d["raw_turn_contains_exact_lexical_sequence"] for d in decisions),
            "selected_exact_sequence_support": sum(d["selected_row_contains_exact_lexical_sequence"] for d in decisions),
            "mean_abs_onset_delta_s": round(sum(abs(d["onset_delta_s"]) for d in supported)/len(supported), 4) if supported else None,
            "mean_abs_end_delta_s": round(sum(abs(d["end_delta_s"]) for d in supported)/len(supported), 4) if supported else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-archive", required=True, type=Path)
    parser.add_argument("--media", type=Path, help="Relocated identical original media; defaults to authenticated archive path")
    parser.add_argument("--model-resources", type=Path)
    parser.add_argument("--asr-resource")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    authority = read_authority(args.diagnostic_archive)
    if args.media:
        path = args.media
    else:
        paths = [Path(p) for p in authority["media_paths"] if Path(p).is_file()]
        if not paths:
            raise FileNotFoundError("Original media relocated; provide --media after bounded discovery")
        path = paths[0]
    media = probe_media(path, audio_stream_index=authority["selected_audio_stream"])
    if (media.fingerprint_sha256 != authority["fingerprint_sha256"]
            or abs(media.duration_s-authority["duration_s"]) > .01):
        raise ValueError("Original media fingerprint/duration mismatch")
    validate_regions(media.duration_s)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = args.output_dir
    plan = {"diagnostic_sha256": DIAGNOSTIC_SHA, "media_fingerprint_sha256": media.fingerprint_sha256,
            "source_language": "fr", "selected_audio_stream": media.selected_audio_stream,
            "duration_s": media.duration_s, "regions": REGIONS, "policies": POLICIES,
            "max_requests": MAX_REQUESTS, "max_inference_audio_s": MAX_INFERENCE_AUDIO_S,
            "reference_tolerance_s": TOLERANCE_S, "acoustic_truth_accepted": False}
    write_json(root/"plan.json", plan)
    if args.plan_only:
        print("Authenticated media and registered matched windows; no inference.")
        return

    resource = load_model_resources(args.model_resources).resolve(
        "asr", resource_name=args.asr_resource, require_local_paths=True)
    if (resource.adapter != "nemotron_ctc" or not resource.checkpoint_path
            or Path(resource.checkpoint_path).resolve() != Path(CHECKPOINT).resolve()
            or not Path(CHECKPOINT).is_dir()):
        raise ValueError("Expected existing exact Nemotron checkpoint/resource; no substitution")
    # Different served aliases are not different weights, but are disclosed.
    # Live model/quantization/CTC ownership is independently checked on host.
    plan["resource"] = {"name": resource.name, "fingerprint": resource.fingerprint(),
                        "model": resource.model, "historical_served_models": authority["served_models"]}
    write_json(root/"plan.json", plan)
    jobs = {}
    inputs = root/"inputs"
    inputs.mkdir()
    for name, region, target, probes in REGIONS:
        for label, crop in (("region", region), ("local_a", probes[0]), ("local_b", probes[1])):
            wav = inputs/f"{name}-{label}.wav"
            extract_wav_segment(media.path, start_s=crop[0], duration_s=crop[1]-crop[0],
                                out_wav=wav, audio_stream_index=media.selected_audio_stream)
            duration, digest = validate_wav_file(wav)
            if abs(duration-(crop[1]-crop[0])) > .05:
                raise ValueError("Matched input extraction duration mismatch")
            jobs[name, label] = (wav, digest, crop)
    client = NemotronVLLMClient(base_url=resource.endpoint, model=resource.model,
                               api_key=resource.credential(), timeout_s=900.)
    summary, private = {}, {}
    write_json(root/"summary.json", {"status": "in_progress", "plan": plan, "regions": summary})
    with managed_asr_client(ModelLifecycle(), resource, client) as stage:
        bounded = BudgetedClient(stage, root/"private-requests.json")
        for name, region, target, probes in REGIONS:
            metrics, observations = {}, {}
            # Alternate policy execution order to avoid consistently favoring
            # the second policy in warm-cache wall-time comparisons.
            order = ("fixed", "pause_aware") if REGIONS.index((name,region,target,probes)) % 2 == 0 else ("pause_aware", "fixed")
            for label in (*order, "local_a", "local_b"):
                wav, digest, crop = jobs[name, label if label.startswith("local_") else "region"]
                if sha_file(wav) != digest:
                    raise ValueError("Matched audio changed before request")
                run = root/"runs"/name/label
                bounded.run = f"{name}/{label}"
                before = len(bounded.requests)
                policy = POLICIES.get(label, POLICIES["fixed"])
                cfg = RecordingPipelineConfig(source_language="fr", mode="speaker",
                    audio_stream_index=0, speaker_embeddings="none", alignment_fallback="off",
                    resume=False, model_path=resource.checkpoint_path,
                    asr_resource_fingerprint=resource.fingerprint(), **policy)
                out = process_media(wav, output_dir=run, client=bounded, config=cfg)
                m, obs = inspect_run(out, crop, digest)
                # Namespace matching IDs across all jobs without changing
                # actual recording/source identities inside their contracts.
                for row in obs["all_rows"]:
                    row["observation_id"] = bounded.run + ":" + row["id"]
                request_rows = bounded.requests[before:]
                m["actual_requests"] = len(request_rows)
                m["actual_inference_audio_s"] = round(sum(r["audio_seconds"] for r in request_rows),4)
                m["request_wall_s"] = round(sum(r.get("wall_s",0) for r in request_rows),4)
                metrics[label], observations[label] = m, obs
            reference = corroborated_events(observations["local_a"], observations["local_b"],
                                            target, TOLERANCE_S)
            comparisons = {label: compare_reference(reference, observations[label]) for label in POLICIES}
            summary[name] = {"region": list(region), "target": list(target), "runs": metrics,
                             "same_model_reference_events": len(reference),
                             "comparisons": {label: comparison_metrics(ds) for label,ds in comparisons.items()},
                             "acoustic_truth_accepted": False}
            private[name] = {"reference_events": reference, "comparisons": comparisons}
            write_json(root/"private-comparison.json", private)
            write_json(root/"summary.json", {"status": "in_progress", "plan": plan, "regions": summary})
        totals = {"actual_requests": len(bounded.requests),
                  "actual_inference_audio_s": round(sum(r["audio_seconds"] for r in bounded.requests),4)}
    write_json(root/"summary.json", {"status": "completed", "plan": plan, "regions": summary,
                                     "totals": totals,
                                     "limitations": [
                                         "Same-model positive support, not independent acoustic or human truth.",
                                         "No strict event match is not proof of omission; timing deltas are relative to observed hypotheses.",
                                         "Region-local planning; not full-episode coverage or exact historical executable replay.",
                                         "No default promotion or canonical recovery decision made by this evaluator."
                                     ]})
    print("Completed bounded matched geometry evaluation; return private comparison and source evidence.")


if __name__ == "__main__":
    main()
