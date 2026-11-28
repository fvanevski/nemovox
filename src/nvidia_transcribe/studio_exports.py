"""Explicit host-output exports for local single-user Studio jobs.

Full internal job artifacts remain authoritative. Exports are bounded copies
into configured host-owned output roots, never moves or arbitrary host writes.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

ARTIFACT_PATHS = {
    "source_transcript": "asr/transcript.txt",
    "recording_contract": "asr/recording.contract.json",
    "refined_contract": "output/refined.contract.json",
    "translated_contract": "output/translated.contract.json",
    "raw_source_transcript": "output/raw_source_transcript.txt",
    "refined_source_transcript": "output/refined_source_transcript.txt",
    "final_english": "output/final_english_subtitles.txt",
    "subtitle_metadata": "output/subtitles.metadata.json",
    "presentation_observations": "output/presentation.observations.json",
}
EXPORT_TYPES = frozenset((*ARTIFACT_PATHS, "srt", "ass"))


def export_roots(project: Path, configured: str | None = None) -> tuple[Path, ...]:
    """Roots are explicit host configuration, never the media-read allowlist."""
    raw = configured if configured is not None else os.getenv("NEMOVOX_EXPORT_ROOTS", "")
    values = [part for part in raw.split(os.pathsep) if part] if raw else []
    if not values:
        values = [str(project / "outputs" / "exports")]
    return tuple(dict.fromkeys(Path(v).expanduser().resolve() for v in values))


def validate_destination(value: str, roots: tuple[Path, ...]) -> Path:
    if not value or "\x00" in value or not Path(value).expanduser().is_absolute():
        raise ValueError("Select an absolute host output directory.")
    target = Path(value).expanduser().resolve()
    if not any(target.is_relative_to(root.resolve()) for root in roots):
        raise ValueError("Output directory is outside configured writable export roots.")
    if target.exists() and not target.is_dir():
        raise ValueError("Output destination exists but is not a directory.")
    return target


def browse_destinations(roots: tuple[Path, ...], path: str | None = None) -> dict[str, Any]:
    if not roots:
        raise ValueError("No configured output roots.")
    current = validate_destination(path or str(roots[0]), roots)
    children: list[dict[str, str]] = []
    if current.is_dir():
        for child in sorted(current.iterdir(), key=lambda p: p.name.casefold())[:250]:
            # Resolve each child before showing it: omit symlinks escaping roots.
            if child.is_dir():
                try:
                    checked = validate_destination(str(child), roots)
                except ValueError:
                    continue
                children.append({"name": child.name, "path": str(checked)})
    parent = current.parent
    return {
        "roots": [str(root) for root in roots],
        "default": str(roots[0]),
        "directory": str(current),
        "parent": str(parent) if parent != current and
            any(parent.is_relative_to(root) for root in roots) else None,
        "directories": children,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_no_overwrite(source: Path, destination: Path) -> str:
    """Atomically publish without replacing a different existing file."""
    if not source.is_file() or source.is_symlink():
        raise ValueError("Internal source artifact is missing or unsafe.")
    expected = _sha256(source)
    if destination.exists() or destination.is_symlink():
        if destination.is_file() and not destination.is_symlink() and _sha256(destination) == expected:
            return "already_saved"
        raise FileExistsError("A different export file already exists.")
    tmp = destination.with_name("." + destination.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with source.open("rb") as inp, tmp.open("xb") as out:
            shutil.copyfileobj(inp, out, length=1024 * 1024)
        if _sha256(tmp) != expected:
            raise OSError("Export checksum mismatch.")
        os.link(tmp, destination)  # EEXIST is safe; never use os.replace here.
        return "saved"
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_report(path: Path, report: dict[str, Any]) -> None:
    tmp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def export_selected(
    job_root: Path,
    job_id: str,
    plan: dict[str, Any] | None,
    roots: tuple[Path, ...],
    *,
    remote_media: Path | None = None,
    media_only: bool = False,
) -> dict[str, Any]:
    """Persist per-item results; remote copies can survive a later ASR failure.

    A fixed per-job child directory avoids cross-job naming collisions. Retrying
    a completed copy is idempotent by file digest; conflicting files fail closed.
    """
    report_path = job_root / "exports.json"
    if not plan or not plan.get("destination"):
        return {"status": "disabled", "directory": None, "items": []}
    prior: dict[str, Any] = {}
    if report_path.is_file():
        try:
            prior = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prior = {}
    entries: dict[str, dict[str, Any]] = {
        item["type"]: item for item in prior.get("items", [])
        if isinstance(item, dict) and isinstance(item.get("type"), str)
    }
    destination_base = validate_destination(str(plan["destination"]), roots)
    destination = destination_base / ("nemovox-" + job_id)
    remote_requested = ["remote_media"] if (
        plan.get("save_remote_media") and remote_media is not None) else []
    # Recheck remote export on finalization, in case the initial copy failed.
    requested = remote_requested if media_only else list(plan.get("artifacts", [])) + remote_requested
    for kind in requested:
        if kind not in EXPORT_TYPES and kind != "remote_media":
            entries[kind] = {"type": kind, "status": "failed", "error": "Unknown artifact type."}
            continue
        candidates = sorted((job_root / "output").glob("*." + kind)) if kind in ("srt", "ass") \
            else [job_root / ARTIFACT_PATHS[kind]] if kind in ARTIFACT_PATHS \
            else [remote_media] if remote_media is not None else []
        # Imported recording contracts skip process-media, but the subtitle
        # pipeline still writes the unchanged source-language transcript here.
        if kind == "source_transcript" and not any(p.is_file() for p in candidates):
            candidates = [job_root / "output/raw_source_transcript.txt"]
        if kind == "remote_media" and not plan.get("save_remote_media"):
            continue
        if not candidates or not candidates[0] or not candidates[0].is_file():
            entries[kind] = {"type": kind, "status": "failed", "error": "Requested artifact not generated."}
            continue
        source = candidates[0]
        file_name = ("remote-media" + source.suffix.lower()) if kind == "remote_media" else source.name
        try:
            # Revalidate after directory creation; symlink movement may change the target.
            destination_base.mkdir(parents=True, exist_ok=True)
            validate_destination(str(destination_base), roots)
            destination.mkdir(exist_ok=True)
            if destination.is_symlink() or not destination.resolve().is_relative_to(destination_base.resolve()):
                raise ValueError("Unsafe export job directory.")
            target = destination / file_name
            status = _copy_no_overwrite(source, target)
            entries[kind] = {
                "type": kind, "status": status, "source": str(source.relative_to(job_root)),
                "destination": str(target), "sha256": _sha256(source),
            }
        except (OSError, ValueError) as exc:
            entries[kind] = {"type": kind, "status": "failed", "error": str(exc)}
    failures = [item for item in entries.values() if item.get("status") == "failed"]
    saved = [item for item in entries.values() if item.get("status") in ("saved", "already_saved")]
    report = {
        "status": "partial" if saved and failures else "failed" if failures else
                  "complete" if entries else "pending",
        "directory": str(destination), "items": list(entries.values()),
        "updated_at": time.time(),
    }
    _atomic_report(report_path, report)
    return report
