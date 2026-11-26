"""Read-only verification of current rendered output receipts."""
import hashlib
import json
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def current_render_metadata(job, translated):
    # A failed review alone is not a render. Admit an explicit offline
    # salvage receipt only for these exact contract and subtitle bytes.
    review = translated.get("translation_review") or {}
    failed = review.get("status") == "completed_with_errors"
    path = job / "subtitles.metadata.json"
    if failed:
        try:
            metadata = read(path)
            retained = metadata["translation_review"]
            receipt = metadata["salvage_render_receipt"]
            contract_path = job / "translated.contract.json"
            if (metadata["render_status"] != "rendered_with_review_errors"
                    or retained["status"] != "reused_translated_contract"
                    or retained["retained_review_status"] != "completed_with_errors"
                    or retained["errors"] != review.get("errors", [])
                    or retained["resource_blocked_reason"] != review.get("resource_blocked_reason")
                    or receipt["schema"] != "nemovox-failed-review-render/v1"
                    or read(contract_path) != translated
                    or receipt["translated_contract_sha256"] != sha(contract_path)):
                return None
            outputs = receipt["outputs"]
            expected = {fmt for fmt in ("srt", "ass") if fmt in metadata["files"]}
            if not expected or set(outputs) != expected:
                return None
            for fmt, item in outputs.items():
                name = item["name"]
                if (not isinstance(name, str) or Path(name).name != name
                        or Path(name).suffix != "." + fmt
                        or Path(metadata["files"][fmt]).name != name
                        or sha(job / name) != item["sha256"]):
                    return None
            if not isinstance(metadata["subtitles"], dict):
                return None
            return metadata
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return None
    if not path.is_file():
        raise ValueError("Missing render metadata without explicit failed review")
    metadata = read(path)
    if metadata.get("render_status") == "failed_no_renderable_dialogue":
        return None
    if metadata.get("render_status") == "rendered_with_review_errors" or "salvage_render_receipt" in metadata:
        raise ValueError("Stale failed-review salvage metadata for nonfailed contract")
    return metadata
