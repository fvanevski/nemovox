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


def read_render_manifest(output: Path) -> dict:
    """Read the job's current output manifest, not a filename glob.

    An absent manifest permits the single-file retained-job fallback. A
    malformed manifest does not establish any current render artifacts.
    """
    path = output / "subtitles.metadata.json"
    if not path.is_file():
        return {}
    try:
        metadata = read(path)
    except (OSError, ValueError):
        return {"files": {}}
    return metadata if isinstance(metadata, dict) else {"files": {}}


def rendered_artifact_paths(output: Path, metadata: dict) -> dict[str, Path]:
    """Resolve only current emitted files within the job's output directory.

    Retained job directories can move; manifest basenames remain meaningful,
    whereas the original absolute path must never be dereferenced. An absent
    format in a current manifest is not permission to select an older file.
    Pre-manifest jobs may expose an unambiguous subtitle file without claiming
    that it belongs to current-format lineage.
    """
    result = {}
    if "files" in metadata or "presentation" in metadata:
        files = metadata.get("files")
        if not isinstance(files, dict):
            return result
        for kind in ("srt", "ass", "presentation_observations"):
            value = files.get(kind)
            if not isinstance(value, str) or not value:
                continue
            path = output / Path(value).name
            matches = (path.name == "presentation.observations.json"
                       if kind == "presentation_observations"
                       else path.suffix == "." + kind)
            if matches and path.is_file() and not path.is_symlink():
                result[kind] = path
        return result
    for kind in ("srt", "ass"):
        paths = [p for p in output.glob("*." + kind)
                 if p.is_file() and not p.is_symlink()]
        if len(paths) == 1:
            result[kind] = paths[0]
    return result
