"""Offline qualification of the anchor-only review change on sealed proposals.

Run this identical script with the baseline and candidate source imports.
Never mutate/rekey archived checkpoints; require every native proposal and
comparison to be a hit of the original evidence-bound key.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

from nvidia_transcribe import translation_review as review
from nvidia_transcribe.refinement import GlossaryConfig

ARCHIVE_SHA = "6f3f0e308b227e69248e4121f9064c5e7e3e9bb543a6fe554d7e6426866a592a"
CASES = {"bounded": 26, "mandarin": 2, "french": 221}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_archive(path):
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != ARCHIVE_SHA:
            raise ValueError("Wrong completion archive")
    entries = {}
    with tarfile.open(path, "r:gz") as tar:
        for entry in tar:
            name = PurePosixPath(entry.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("Unsafe archive path")
            if entry.isdir():
                continue
            if not entry.isfile() or entry.size > 64 * 1024 * 1024 or str(name) in entries:
                raise ValueError("Unexpected archive member")
            entries[str(name)] = tar.extractfile(entry).read()
    manifest = {}
    for line in entries.pop("MANIFEST.sha256").decode().splitlines():
        digest, name = line.split("  ", 1)
        if name in manifest or name not in entries or sha(entries[name]) != digest:
            raise ValueError("Invalid manifest member: " + name)
        manifest[name] = digest
    if len(manifest) != 96 or set(manifest) != set(entries):
        raise ValueError("Archive inventory mismatch")
    return entries


class NoInference(BaseException):
    pass


class CacheOnly:
    def complete(self, **kwargs):
        raise NoInference("Review cache miss; no inference authorized")


def replay(entries, case, outdir, baseline):
    prefix = f"jobs/{case}/"
    original = json.loads(entries[prefix + "translated.contract.json"])
    prior_report = original["translation_review"]
    policy = prior_report["review_policy"]
    resources = json.loads(entries["config/candidate-resources.json"])
    resource = resources["resources"][resources["roles"]["translation_review"]]
    gl = GlossaryConfig()
    if (prior_report["status"] != "completed" or prior_report["subdivisions"]
            or prior_report["glossary_digest"] != gl.digest()
            or len(original["utterances"]) != CASES[case]):
        raise ValueError("Unexpected retained review authority")
    cfg = review.TranslationReviewConfig(
        profile=prior_report["profile"], style=prior_report["style"],
        target_language=policy["target_language"],
        context_utterances_before=policy["context_before"],
        context_utterances_after=policy["context_after"], batch_size=prior_report["batch_size"],
        base_url=resource["endpoint"], model=resource["model"],
        resource_fingerprint=prior_report["resource_fingerprint"],
        max_tokens=policy["max_tokens"], temperature=policy["temperature"],
        reasoning_effort=policy["reasoning_effort"], wire_protocol=policy["wire_protocol"],
        thinking_budget=policy["thinking_budget"], use_json_schema=True)
    outdir.mkdir(parents=True, exist_ok=False)
    checkpoint = outdir / "translation-review.checkpoint.json"
    checkpoint.write_bytes(entries[prefix + "translation-review.checkpoint.json"])
    checkpoint_hash = sha(checkpoint.read_bytes())
    rows, report = review.review_translated_utterances(
        original["utterances"], source_language=original["source_language"],
        config=cfg, glossary=gl, text_client=CacheOnly(), checkpoint_path=checkpoint)
    if (report["status"] != "completed" or report["cache_hits"] != CASES[case]
            or report["api_calls"] != 0 or sha(checkpoint.read_bytes()) != checkpoint_hash):
        raise ValueError("Incomplete or mutated cache replay")
    decisions = []
    for a, b in zip(original["utterances"], rows, strict=True):
        # Review resets these fields; all other lineage and MT must be exact.
        def invariant(row):
            return {k:v for k,v in row.items() if k != "translated_text"
                    and not k.startswith(("translation_review", "translation_comparison"))
                    and k != "translation_comparative_review"}
        if invariant(a) != invariant(b):
            raise ValueError("Non-review row mutation: " + a["id"])
        for key in ("translation_reviewed_text", "translation_review_reason",
                    "translation_review_source_sha256", "translation_review_batch_sha256",
                    "translation_comparative_review", "translation_review_passage"):
            if a.get(key) != b.get(key):
                raise ValueError("Proposal or decision provenance mismatch: " + a["id"] + " " + key)
        old, new = a["translation_review_status"], b["translation_review_status"]
        if baseline and (old != new or a["translated_text"] != b["translated_text"]
                         or a.get("translation_review_rejection_reason") != b.get("translation_review_rejection_reason")):
            raise ValueError("Baseline replay differs: " + a["id"])
        if old != new or a["translated_text"] != b["translated_text"]:
            if (old not in ("review_rejected_unresolved_source_preserved_machine_translation",
                            "review_rejected_dependent_passage_preserved_machine_translation")
                    or new != "corrected"
                    or a.get("translation_review_rejection_reason") == "unresolved_source_material_lexical_change"):
                raise ValueError("Unexpected changed disposition: " + a["id"])
            decisions.append({"id": a["id"], "source": a["source_text"],
                              "primary": a["machine_translated_text"],
                              "prior": a["translated_text"], "candidate": b["translated_text"],
                              "prior_status": old, "candidate_status": new,
                              "uncertainty": a.get("review_source_uncertainty"),
                              "comparison": a.get("translation_comparative_review"),
                              "passage": a.get("translation_review_passage")})
    if baseline:
        for field in ("reviewed", "changed", "rejected_polarity", "comparative_rejections",
                      "dependent_passage_rejections", "joined_passage_rejections",
                      "unresolved_source_rejections", "direction_grammar_rejections"):
            if report[field] != prior_report[field]:
                raise ValueError("Baseline metric mismatch: " + field)
    if case == "mandarin" and decisions:
        raise ValueError("Mandarin protection changed")
    if not baseline and case == "bounded":
        changed_ids = {x["id"] for x in decisions}
        if original["utterances"][19]["id"] not in changed_ids:
            raise ValueError("Demonstrated anchor failure not repaired")
        if not any("headed straight west" in x["candidate"].casefold() for x in decisions):
            raise ValueError("Joined westward meaning not restored")
    # A distinct offline derived contract, not a new live inference receipt.
    derived = dict(original, utterances=rows, translation_review=report)
    (outdir / "translated.contract.json").write_text(json.dumps(derived, ensure_ascii=False, indent=2))
    (outdir / "private-delta.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2))
    summary = {"case": case, "baseline": baseline, "rows": len(rows),
               "cache_hits": report["cache_hits"], "api_calls": report["api_calls"],
               "changed_dispositions": len(decisions), "checkpoint_sha256": checkpoint_hash,
               "review": {k: report[k] for k in ("status", "reviewed", "changed",
                  "rejected_polarity", "comparative_rejections", "dependent_passage_rejections",
                  "joined_passage_rejections", "unresolved_source_rejections",
                  "direction_grammar_rejections")}}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    if not Path(review.__file__).resolve().is_relative_to((args.source_root / "src").resolve()):
        raise ValueError("Wrong review import binding")
    entries = load_archive(args.archive)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for case in CASES:
        print(json.dumps(replay(entries, case, args.output_dir / case, args.baseline)))


if __name__ == "__main__":
    main()
