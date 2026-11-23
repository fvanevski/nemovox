"""Explicit offline reviewed-source revision to private SRT/ASS outputs.

No model calls, active-job writes, sidecar export, automatic component selection,
or implicit conversion of uncertain observations into acoustic facts.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from nvidia_transcribe.evidence_fusion import _digest, plan_reviewed_source_partitions
from nvidia_transcribe.source_fusion_application import build_reviewed_source_revision
from nvidia_transcribe.source_fusion_downstream import (
    assemble_revised_translation, attach_local_acoustic_voices,
)
from nvidia_transcribe.subtitles import SubtitleConfig, render_subtitles

SELECTION_SCHEMA = "nemovox-reviewed-component-selection/v1"
REPORT_SCHEMA = "nemovox-reviewed-revision-run/v1"


def _read_json(path: str | Path) -> tuple[Any, str]:
    source = Path(path)
    raw = source.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _write_new_json(root: Path, name: str, value: Any) -> str:
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    dest = root / name
    with dest.open("xb") as out:
        out.write(encoded)
    return hashlib.sha256(encoded).hexdigest()


def run_reviewed_revision(
    *,
    original_recording_path: str | Path,
    source_evidence_path: str | Path,
    review_dossier_path: str | Path,
    selection_path: str | Path,
    previous_translated_path: str | Path | None = None,
    translation_receipts_path: str | Path | None = None,
    local_voice_review_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    episode_stem: str | None = None,
    formats: tuple[str, ...] = ("srt", "ass"),
    apply: bool = False,
    subtitle_config: SubtitleConfig | None = None,
) -> dict[str, Any]:
    """Preview or execute a reviewed, complete, opt-in source revision.

    Selection carries an exact plan SHA and input content digests. The source,
    evidence, and dossier are revalidated on every invocation. All outputs
    are staged in a new directory and published only after a valid render.
    """
    if not isinstance(apply, bool):
        raise ValueError("apply must be an explicit boolean")
    original, original_file_sha = _read_json(original_recording_path)
    ledger, ledger_file_sha = _read_json(source_evidence_path)
    dossier, dossier_file_sha = _read_json(review_dossier_path)
    selection, selection_file_sha = _read_json(selection_path)
    if not isinstance(selection, dict) or selection.get("schema") != SELECTION_SCHEMA:
        raise ValueError("Unsupported reviewed component selection schema")
    plan = plan_reviewed_source_partitions(original, ledger, dossier)
    plan_sha = _digest(plan)
    selected = selection.get("selected_component_ids")
    if (not isinstance(selected, list) or not selected or any(
        not isinstance(item, str) or not item for item in selected
    ) or len(set(selected)) != len(selected)):
        raise ValueError("Explicit, nonempty, distinct reviewed components required")
    if selection.get("expected_plan_sha256") != plan_sha:
        raise ValueError("Reviewed component selection references a stale plan")
    for name, doc in (("recording", original), ("ledger", ledger), ("dossier", dossier)):
        if selection.get(f"{name}_content_sha256") != _digest(doc):
            raise ValueError(f"Reviewed component selection has stale {name} identity")
    selected_set = set(selected)
    ready = {item["id"] for item in plan["connected_components"] if
             item["status"] == "complete_reviewed_component_requires_application_gate"}
    if not selected_set <= ready:
        raise ValueError("Selected components contain incomplete or unreviewed evidence")
    preview = build_reviewed_source_revision(
        original, ledger, dossier, selected_component_ids=selected,
        expected_plan_sha256=plan_sha, enable=False,
    )
    if preview["new_recording"] is not None:
        raise ValueError("Disabled reviewed revision unexpectedly returned a source")
    summary: dict[str, Any] = {
        "schema": REPORT_SCHEMA, "status": "preview", "applied": False,
        "source_input_sha256": {
            "recording_file": original_file_sha, "ledger_file": ledger_file_sha,
            "dossier_file": dossier_file_sha, "selection_file": selection_file_sha,
        },
        "partition_plan_sha256": plan_sha,
        "selected_component_ids": sorted(selected),
        "unselected_blocked_components": [c["id"] for c in plan["connected_components"]
                                          if c["status"] != "complete_reviewed_component_requires_application_gate"],
        "source_revision_preview": preview["report"],
        "no_model_inference": True, "no_original_input_mutation": True,
    }
    if not apply:
        return summary
    if (not previous_translated_path or not translation_receipts_path or
            not output_dir or not isinstance(episode_stem, str) or not episode_stem.strip()):
        raise ValueError("Application needs prior translation, fresh receipts, output dir and episode stem")
    if episode_stem in (".", "..") or "/" in episode_stem or "\\" in episode_stem:
        raise ValueError("Episode stem must be a filename component")
    if not formats or set(formats) - {"srt", "ass"} or len(formats) != len(set(formats)):
        raise ValueError("Output formats must be distinct SRT/ASS names")
    destination = Path(output_dir).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Reviewed revision output directory already exists")
    if not destination.parent.is_dir():
        raise FileNotFoundError("Reviewed output parent directory must already exist")
    result = build_reviewed_source_revision(
        original, ledger, dossier, selected_component_ids=selected,
        expected_plan_sha256=plan_sha, enable=True,
    )
    reviewed = result["new_recording"]
    if reviewed is None:
        raise ValueError("Activated source revision did not return a recording")
    if local_voice_review_path is not None:
        voices, voice_sha = _read_json(local_voice_review_path)
        reviewed = attach_local_acoustic_voices(reviewed, dossier, voices)
        summary["source_input_sha256"]["local_voice_review_file"] = voice_sha
    prior, prior_sha = _read_json(previous_translated_path)
    receipts_doc, receipts_sha = _read_json(translation_receipts_path)
    if isinstance(receipts_doc, dict):
        receipts = receipts_doc.get("receipts")
    else:
        receipts = receipts_doc
    if not isinstance(receipts, list):
        raise ValueError("Translation receipts must be a list or a package containing receipts")
    preflight = assemble_revised_translation(original, prior, reviewed, receipts, enable=False)
    if preflight["report"]["missing_fresh_translation_ids"]:
        raise ValueError("Fresh revised-source translation receipts are missing")
    assembled = assemble_revised_translation(original, prior, reviewed, receipts, enable=True)
    translated = assembled["translated_contract"]
    if translated is None or not isinstance(translated.get("utterances"), list):
        raise ValueError("No authenticated revised translated contract")
    summary["source_input_sha256"].update({
        "previous_translated_file": prior_sha, "translation_receipts_file": receipts_sha,
    })
    summary["source_revision_report"] = result["report"]
    summary["translation_bridge_report"] = assembled["report"]
    summary["translated_utterances"] = len(translated["utterances"])
    staging = Path(tempfile.mkdtemp(prefix=".reviewed-revision-", dir=destination.parent))
    try:
        file_sha = {
            "reviewed.recording.contract.json": _write_new_json(staging, "reviewed.recording.contract.json", reviewed),
            "reviewed.translated.contract.json": _write_new_json(staging, "reviewed.translated.contract.json", translated),
        }
        rendered = render_subtitles(
            translated, output_dir=staging, episode_stem=episode_stem,
            target_language=str(translated.get("target_language") or "en"),
            formats=formats, config=subtitle_config or SubtitleConfig(),
            sidecar_dir=None, overwrite_sidecar=False,
        )
        for fmt, filename in rendered["outputs"].items():
            file_sha[Path(filename).name] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
        summary["status"] = "applied_private_output"
        summary["applied"] = True
        summary["render_stats"] = rendered["stats"]
        summary["output_sha256"] = file_sha
        _write_new_json(staging, "reviewed.revision.report.json", summary)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Output directory appeared during reviewed render")
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return summary
