#!/usr/bin/env python3
"""Stage, package, and verify the transcription-reliability campaign evidence archive."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path("/workspace/nvidia-transcribe")
BASE_SHA = "889c28189e9418d45cf5de87b979135c5c4dc3c2"
STAGE = ROOT / "outputs/handbacks/_staging_reliability"
ARCHIVE = ROOT / "outputs/handbacks/transcription-reliability-evidence.tar.gz"
MANIFEST_OUT = ROOT / "evaluations/transcription-reliability/artifact-manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_item(rel_src: str, rel_dst: str | None = None) -> None:
    src = ROOT / rel_src
    dst = STAGE / (rel_dst or rel_src)
    if not src.exists():
        raise FileNotFoundError(f"Missing required evidence artifact: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def bounded_contract_excerpt(src_rel: str, dst_rel: str, max_utterances: int = 60, max_chunks: int = 16) -> None:
    src = ROOT / src_rel
    if not src.is_file():
        return
    data = json.loads(src.read_text(encoding="utf-8"))
    excerpt = {
        "schema_version": data.get("schema_version"),
        "source": data.get("source"),
        "source_language": data.get("source_language"),
        "language_provenance": data.get("language_provenance"),
        "model_provenance": data.get("model_provenance"),
        "processing": data.get("processing"),
        "speakers": data.get("speakers"),
        "speaker_reconciliation": data.get("speaker_reconciliation"),
        "alignment_diagnostics": data.get("alignment_diagnostics"),
        "processing_warnings": data.get("processing_warnings"),
        "utterances_total": len(data.get("utterances") or []),
        "utterances_excerpt": (data.get("utterances") or [])[:max_utterances],
    }
    dst = STAGE / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(excerpt, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True, exist_ok=True)
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)

    # 1. Core configs, docs, scripts, and all campaign reports
    for rel in (
        "AGENTS.md",
        "README.md",
        "pyproject.toml",
        ".gitignore",
        "configs/vllm_rtx5090.env",
        "scripts/start_vllm_server.sh",
        "evaluations/transcription-reliability/HANDback.md",
        "evaluations/transcription-reliability/architecture.md",
        "evaluations/transcription-reliability/correctness-regressions.md",
        "evaluations/transcription-reliability/alignment-fallback.md",
        "evaluations/transcription-reliability/refinement-evaluation.md",
        "evaluations/transcription-reliability/diarization-evaluation.md",
        "evaluations/transcription-reliability/test-results.md",
        "evaluations/transcription-reliability/benchmark.json",
    ):
        copy_item(rel)

    # 2. All production source, reliability scripts, and CPU tests
    for folder in ("src/nvidia_transcribe", "tests", "experiments/reliability", "experiments/modular/tests"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                copy_item(str(path.relative_to(ROOT)))

    # 3. Deterministic manifests & compact JSON run outputs (exclude large WAV/WEBM/logs/weights)
    for rel in (
        "outputs/audio/zh/zh_manifest.json",
        "outputs/audio/overlap/fr_fleurs_manifest.json",
        "outputs/audio/overlap/overlap_manifest.json",
        "outputs/reliability_runs/phase_c_refinement.json",
        "outputs/reliability_runs/phase_d_scene_diarization.json",
        "outputs/reliability_runs/combined_acceptance_summary.json",
        "outputs/reliability_runs/fr_full_episode_repaired_90s/transcript.txt",
        "outputs/chunked_runs/fr_full_episode_90s/transcript.txt",
    ):
        if (ROOT / rel).is_file():
            copy_item(rel)

    for run_name in (
        "zh_fleurs_01_2006_off",
        "zh_fleurs_01_2006_qwen",
        "zh_fleurs_05_1805_qwen",
        "zh_2spk_overlap_qwen",
        "zh_multichunk_synthetic_off",
        "zh_multichunk_synthetic_qwen",
        "fr_child_voices_f1_qwen",
    ):
        for fname in ("recording.contract.json", "transcript.txt", "manifest.json"):
            rel = f"outputs/reliability_runs/{run_name}/{fname}"
            if (ROOT / rel).is_file():
                copy_item(rel)

    bounded_contract_excerpt(
        "outputs/reliability_runs/fr_full_episode_repaired_90s/recording.contract.json",
        "outputs/samples/fr_full_episode_repaired_90s.contract.excerpt.json",
        max_utterances=60,
    )
    bounded_contract_excerpt(
        "outputs/chunked_runs/fr_full_episode_90s/recording.contract.json",
        "outputs/samples/fr_full_episode_90s_prior.contract.excerpt.json",
        max_utterances=60,
    )

    # 4. Git provenance & diff since starting baseline HEAD
    head_sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    pyannote_sha = subprocess.check_output(
        ["git", "-C", "/workspace/pyannote", "rev-parse", "HEAD"], text=True
    ).strip()
    git_diff = subprocess.check_output(
        ["git", "-C", str(ROOT), "diff", f"{BASE_SHA}..HEAD"],
        text=True,
        errors="replace",
    )
    git_log = subprocess.check_output(
        ["git", "-C", str(ROOT), "log", f"{BASE_SHA}..HEAD", "--stat"],
        text=True,
    )
    (STAGE / "evaluations/transcription-reliability/git_provenance.json").write_text(
        json.dumps(
            {
                "base_head": BASE_SHA,
                "current_head": head_sha,
                "pyannote_reference_head": pyannote_sha,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (STAGE / "evaluations/transcription-reliability/git_diff_since_base.patch").write_text(git_diff, encoding="utf-8")
    (STAGE / "evaluations/transcription-reliability/git_log_since_base.txt").write_text(git_log, encoding="utf-8")

    # 5. Build SHA-256 manifest
    entries = []
    for p in sorted(STAGE.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(STAGE))
            entries.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha256_file(p)})

    manifest = {
        "archive": str(ARCHIVE),
        "base_head": BASE_SHA,
        "current_head": head_sha,
        "file_count": len(entries),
        "total_staged_bytes": sum(e["bytes"] for e in entries),
        "files": entries,
    }
    manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2)
    MANIFEST_OUT.write_text(manifest_json, encoding="utf-8")
    (STAGE / "evaluations/transcription-reliability/artifact-manifest.json").write_text(manifest_json, encoding="utf-8")

    # 6. Build tar.gz and verify
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with tarfile.open(ARCHIVE, "w:gz") as tf:
        for p in sorted(STAGE.rglob("*")):
            tf.add(p, arcname=str(p.relative_to(STAGE)), recursive=False)

    subprocess.run(["gzip", "-t", str(ARCHIVE)], check=True)
    listed = set(subprocess.check_output(["tar", "-tzf", str(ARCHIVE)], text=True).splitlines())

    for e in entries:
        assert e["path"] in listed, f"Missing from tarball: {e['path']}"
        assert sha256_file(STAGE / e["path"]) == e["sha256"], f"SHA mismatch: {e['path']}"

    shutil.rmtree(STAGE)
    print(f"Verified archive {ARCHIVE} ({ARCHIVE.stat().st_size} bytes, {len(entries)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
