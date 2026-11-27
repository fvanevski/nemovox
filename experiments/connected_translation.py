"""Development-only matched experiment; never replaces the canonical source contract."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time

from nvidia_transcribe.moss_recording import validate_moss_contract
from nvidia_transcribe.model_resources import load_model_resources
from nvidia_transcribe.resource_lifecycle import ModelLifecycle, managed_text_client
from nvidia_transcribe.refinement import load_glossary
from nvidia_transcribe.semantic_context import current_row_context, literal_examples
from nvidia_transcribe.review_uncertainty import bind_review_uncertainty, current_uncertainty
from nvidia_transcribe.translation import RIVA_WORKER_SCRIPT, map_language_for_riva
from nvidia_transcribe.translation_review import TranslationReviewConfig, review_translated_utterances
from nvidia_transcribe.ninfer_messages_client import QUALITY_COMPLETION_TOKENS, QUALITY_TIMEOUT_S
from nvidia_transcribe.subtitles import render_subtitles

VARIANTS = ("strict", "expanded")
TERMINAL = re.compile(r"""[.!?。！？…][\s"'”’」』]*$""")
EMPHATIC = re.compile(r"""[!?！？…][\s"'”’」』]*$""")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def interval(row):
    s, e = row.get("source_start"), row.get("source_end")
    if (type(s) in (int, float) and type(e) in (int, float)
            and math.isfinite(s) and math.isfinite(e) and 0 <= s < e):
        return s, e
    return None


def index(row):
    return (row.get("timestamp_provenance") or {}).get("native_event_index")


def normalize(text):
    return "".join(c.casefold() for c in text if c.isalnum())


def bound_context(row):
    # Do not flatten row-bound evidence into a fabricated unit-level citation.
    # Singletons retain all existing review inputs unchanged.
    return bool(current_row_context(row) or current_uncertainty(row)
                or row.get("semantic_context") or row.get("review_source_uncertainty")
                or row.get("source_lineage") or row.get("source_caption_evidence")
                or row.get("source_semantic_resolution")
                or row.get("source_quality_status"))


def joined_text(rows):
    # Do not add lexical whitespace inside CJK scripts.
    def cjk(ch):
        return "\u3400" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff"
    text = rows[0]["source_text"].strip()
    for row in rows[1:]:
        next_text = row["source_text"].strip()
        separator = "" if cjk(text[-1]) and cjk(next_text[0]) else " "
        text += separator + next_text
    return text


def edge_reason(group, row, all_rows, variant):
    a, b = group[-1], row
    ia, ib = interval(a), interval(b)
    if not a.get("speaker_id") or any(r.get("speaker_id") != row.get("speaker_id") for r in group):
        return "speaker_boundary"
    if ia is None or ib is None:
        return "untimed"
    if type(index(a)) is not int or type(index(b)) is not int or index(b) != index(a) + 1:
        return "nonadjacent_acquisition"
    if any(bound_context(r) for r in [*group, row]):
        return "row_bound_uncertainty"
    if normalize(row["source_text"]) in {normalize(r["source_text"]) for r in group}:
        return "possible_repetition"
    if len(group) >= 3 or len(joined_text([*group, row])) > 240:
        return "unit_size"
    start = interval(group[0])[0]
    if ib[1] - start > 12:
        return "unit_duration"
    gap = ib[0] - ia[1]
    if gap < 0 or gap > (0.35 if variant == "strict" else 0.8):
        return "gap_or_overlap"
    ids = {r["id"] for r in [*group, row]}
    if any(r["id"] not in ids and interval(r)
           and interval(r)[0] < ib[1] and interval(r)[1] > start for r in all_rows):
        return "intervening_or_overlapping_event"
    if EMPHATIC.search(a["source_text"]) or (variant == "strict" and TERMINAL.search(a["source_text"])):
        return "sentence_boundary"
    return "eligible"


def partition(rows, all_rows, variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown grouping candidate")
    groups, decisions = [], []
    for row in rows:
        if not groups:
            groups.append([row])
            continue
        reason = edge_reason(groups[-1], row, all_rows, variant)
        decisions.append({"left": groups[-1][-1]["id"], "right": row["id"], "reason": reason})
        if reason == "eligible":
            groups[-1].append(row)
        else:
            groups.append([row])
    if [r["id"] for g in groups for r in g] != [r["id"] for r in rows]:
        raise ValueError("Partition does not preserve ordered occurrence coverage")
    return groups, decisions


def make_unit(group):
    ids = [r["id"] for r in group]
    return {
        "id": ids[0] if len(ids) == 1 else "unit-" + digest(ids)[:24],
        "occurrence_ids": ids,
        "source_text": joined_text(group) if len(group) > 1 else group[0]["source_text"],
        "speaker_id": group[0]["speaker_id"],
        "supported_intervals": [interval(r) for r in group],
        "constituents": deepcopy(group),
    }


def plan_windows(contract, windows, glossary):
    validate_moss_contract(contract)
    rows = bind_review_uncertainty(contract["utterances"], contract.get("source_quality"))
    positions = {r["id"]: i for i, r in enumerate(rows)}
    pair = map_language_for_riva(contract["source_language"])
    if not pair or pair.startswith("en-"):
        raise ValueError("Experiment requires non-English Riva-supported source")
    requests = {}
    examples = {r["id"]: literal_examples(r, glossary, rows[max(0,i-2):i]+rows[i+1:i+3])
                for i,r in enumerate(rows)}

    def request(text, context):
        payload = {"text": text, "lang_pair": pair, "context_examples": context}
        key = digest(payload)
        requests[key] = {"utterance_id": key, **payload}
        return key

    cases = []
    selected = set()
    for first, last in windows:
        lo, hi = positions[first], positions[last]
        if not 0 <= hi-lo < 40:
            raise ValueError("Window must be forward and contain at most 40 rows")
        subset = rows[lo:hi+1]
        ids = {r["id"] for r in subset}
        if selected & ids or len(selected | ids) > 80:
            raise ValueError("Windows must be disjoint, at most 80 total occurrences")
        selected |= ids
        normal = []
        for row in subset:
            unit = make_unit([row])
            unit["request_id"] = request(row["source_text"], examples[row["id"]])
            normal.append(unit)
        case = {"window": [first,last], "normal": normal, "variants": {}}
        for variant in VARIANTS:
            groups, edges = partition(subset, rows, variant)
            units = []
            for group in groups:
                unit = make_unit(group)
                # Same exact ordered examples for joined and constituent MT.
                # Normal arm separately preserves the production example selection.
                shared = {}
                for row in group:
                    for ex in examples[row["id"]]:
                        shared.setdefault(ex["id"], ex)
                common = list(shared.values())[:4]
                unit["request_id"] = request(unit["source_text"], common)
                unit["matched_constituent_requests"] = [
                    request(row["source_text"], common) for row in group]
                units.append(unit)
            case["variants"][variant] = {"units": units, "edges": edges}
        cases.append(case)
    return {"schema": "connected-translation-experiment/v1", "cases": cases,
            "requests": list(requests.values()), "language": contract["source_language"]}


def run_riva(plan, resource, lifecycle):
    if resource.device != "cuda:0":
        raise ValueError("Experiment requires explicit configured cuda:0; no CPU fallback")
    requests = plan["requests"]
    with lifecycle.worker(resource_name=resource.name):
        proc = subprocess.run(
            [resource.worker_python, str(RIVA_WORKER_SCRIPT)],
            input=json.dumps({"model_path": resource.checkpoint_path, "items": requests,
                              "device": resource.device}, ensure_ascii=False),
            capture_output=True, text=True, timeout=max(180, 15 * len(requests)),
        )
    if proc.returncode:
        raise RuntimeError(f"Riva worker failed: {proc.returncode}: {proc.stderr[-2000:]}")
    reply = json.loads(proc.stdout.strip().splitlines()[-1])
    validate_riva(plan, reply, resource.device)
    return reply


def validate_riva(plan, reply, device):
    expected = {r["utterance_id"]: r for r in plan["requests"]}
    results = {}
    if reply.get("status") != "ok" or reply.get("device_used") != device:
        raise ValueError("Riva status/device mismatch")
    for row in reply["results"]:
        uid = row["utterance_id"]
        if uid not in expected or uid in results:
            raise ValueError("Unexpected or duplicate Riva result ID")
        if not isinstance(row.get("translated_text"), str) or not row["translated_text"].strip():
            raise ValueError("Empty translation")
        if row.get("context_example_ids") != [e["id"] for e in expected[uid]["context_examples"]]:
            raise ValueError("Riva context-example mismatch")
        if type(row.get("input_tokens")) is not int or not 0 < row["input_tokens"] <= 8192:
            raise ValueError("Invalid Riva input token receipt")
        results[uid] = row["translated_text"]
    if results.keys() != expected.keys():
        raise ValueError("Incomplete Riva results")
    return results


def adapter_row(unit, translation):
    # This is a disposable review/render projection, NOT a canonical event.
    group = unit["constituents"]
    if len(group) == 1:
        row = deepcopy(group[0])
    else:
        row = {
            "id": unit["id"], "speaker_id": unit["speaker_id"],
            "source_text": unit["source_text"],
            "source_start": unit["supported_intervals"][0][0],
            "source_end": unit["supported_intervals"][-1][1],
            "timestamp_provenance": {"timing_status": "experimental_display_envelope"},
        }
    return {**row, "grouped_utterance_ids": unit["occurrence_ids"],
            "machine_translated_text": translation, "translated_text": translation}


class TraceClient:
    def __init__(self, client, trace):
        self.client, self.trace = client, trace

    def complete(self, **kwargs):
        # Record private requests/results incrementally, including failures.
        record = {"request": kwargs}
        try:
            result = self.client.complete(**kwargs)
            record["response"] = result
            return result
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            raise
        finally:
            with self.trace.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--window", action="append", required=True, help="FIRST_ID:LAST_ID inclusive")
    parser.add_argument("--output", type=Path, required=True, help="New private directory")
    parser.add_argument("--glossary")
    parser.add_argument("--model-resources")
    parser.add_argument("--lifecycle-config")
    parser.add_argument("--profile", choices=["conservative","balanced","expressive"], default="balanced")
    parser.add_argument("--style", choices=["natural","literal","expressive"], default="natural")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--primary-only", action="store_true")
    args = parser.parse_args()
    raw = args.contract.read_bytes()
    contract = json.loads(raw)
    glossary = load_glossary(args.glossary)
    windows = [tuple(w.split(":")) for w in args.window]
    plan = plan_windows(contract, windows, glossary)
    plan["contract_sha256"] = hashlib.sha256(raw).hexdigest()
    plan["glossary_digest"] = glossary.digest()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    write(args.output / "plan.json", plan)
    if args.plan_only:
        return
    registry = load_model_resources(args.model_resources)
    riva = registry.resolve("riva_translation", require_local_paths=True)
    review = registry.resolve("translation_review")
    lifecycle = ModelLifecycle(config_path=args.lifecycle_config)
    write(args.output / "resources.json", {"riva": riva.fingerprint(),
                                          "review": review.fingerprint()})
    reply = run_riva(plan, riva, lifecycle)
    write(args.output / "riva.json", reply)
    translated = validate_riva(plan, reply, riva.device)
    cfg = TranslationReviewConfig(
        profile=args.profile, style=args.style, base_url=review.endpoint,
        model=review.model, resource_fingerprint=review.fingerprint(),
        wire_protocol="ninfer_messages", thinking_budget=2048, reasoning_effort="xhigh",
        max_tokens=max(4096, QUALITY_COMPLETION_TOKENS), timeout_s=max(180, QUALITY_TIMEOUT_S),
    )
    outputs = []
    for i, case in enumerate(plan["cases"]):
        arms = {"normal": case["normal"],
                **{name: case["variants"][name]["units"] for name in VARIANTS}}
        for arm, units in arms.items():
            out = args.output / f"case-{i+1}-{arm}"
            out.mkdir(mode=0o700)
            primary = [adapter_row(u, translated[u["request_id"]]) for u in units]
            write(out / "primary.json", primary)
            write(out / "matched.json", [
                {"occurrence_ids": u["occurrence_ids"],
                 "joined": translated[u["request_id"]],
                 "separate": [translated[k] for k in u.get("matched_constituent_requests", [])]}
                for u in units])
            result = render_subtitles(
                {"source": contract["source"], "utterances": primary},
                output_dir=out, episode_stem="primary",
            )
            write(out / "primary-render.json", result)
            outputs.append((i, out, primary))
    if args.primary_only:
        return
    # One owned service acquisition for the review campaign, after the Riva
    # worker has exited. Identical arms within a case need not be re-reviewed.
    completed = {}
    with managed_text_client(lifecycle, review, timeout_s=cfg.timeout_s,
                             reasoning_effort=cfg.reasoning_effort,
                             wire_protocol=cfg.wire_protocol) as client:
        for i, out, primary in outputs:
            key = (i, digest(primary))
            if key in completed:
                original, reviewed, summary = completed[key]
                summary = {**summary, "experiment_reused_from": original.name,
                           "experiment_incremental_review_s": 0.0}
            else:
                started = time.monotonic()
                reviewed, summary = review_translated_utterances(
                    primary, source_language=plan["language"], config=cfg, glossary=glossary,
                    text_client=TraceClient(client, out / "review-requests.jsonl"), resume=False)
                summary = {**summary, "experiment_incremental_review_s": time.monotonic()-started}
                if summary.get("status") not in ("completed_with_errors", "completed_with_deferred_comparisons"):
                    completed[key] = (out, reviewed, summary)
            if [r["id"] for r in reviewed] != [r["id"] for r in primary]:
                raise ValueError("Review changed unit identity/order")
            write(out / "reviewed.json", reviewed)
            write(out / "review-summary.json", summary)
            result = render_subtitles(
                {"source": contract["source"], "utterances": reviewed},
                output_dir=out, episode_stem="reviewed",
            )
            write(out / "reviewed-render.json", result)


if __name__ == "__main__":
    main()
