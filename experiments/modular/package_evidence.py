#!/usr/bin/env python3
"""Stage and verify the modular-bakeoff evaluation evidence archive."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path("/workspace/nvidia-transcribe")
PYANNOTE = Path("/workspace/pyannote")
BASE_SHA = "721e73789ed2f575bc7c776665a9b04e9a6a097b"
STAGE = ROOT / "outputs/handbacks/_staging_modular"
ARCHIVE = ROOT / "outputs/handbacks/modular-bakeoff-evidence.tar.gz"
MANIFEST_OUT = ROOT / "evaluations/modular-bakeoff/artifact-manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path, rel_dst: str) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"Missing required evidence file: {src}")
    dst = STAGE / rel_dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True, exist_ok=True)
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)

    # 1. Core project metadata & all evaluations/modular-bakeoff reports
    for rel in (
        "AGENTS.md",
        "README.md",
        "pyproject.toml",
        ".gitignore",
        "evaluations/modular-bakeoff/HANDback.md",
        "evaluations/modular-bakeoff/environment.md",
        "evaluations/modular-bakeoff/model-inventory.md",
        "evaluations/modular-bakeoff/benchmark.json",
        "evaluations/modular-bakeoff/recognition-comparison.md",
        "evaluations/modular-bakeoff/alignment-comparison.md",
        "evaluations/modular-bakeoff/diarization-comparison.md",
        "evaluations/modular-bakeoff/integration-assessment.md",
        "evaluations/modular-bakeoff/architecture-decision.md",
        "evaluations/modular-bakeoff/test-results.md",
    ):
        copy_file(ROOT / rel, rel)

    # 2. All experimental source & CPU tests under experiments/modular/
    for p in sorted((ROOT / "experiments/modular").rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            copy_file(p, str(p.relative_to(ROOT)))

    # 3. Logs, dependency freezes, model inventory, raw model outputs & bounded excerpts
    for rel in (
        "outputs/modular-eval/logs/pip_freeze_asr.txt",
        "outputs/modular-eval/logs/pip_freeze_diar.txt",
        "outputs/modular-eval/logs/model_inventory_raw.json",
        "outputs/modular-eval/logs/pytest_prod.txt",
        "outputs/modular-eval/logs/pytest_modular.txt",
        "outputs/modular-eval/logs/run_asr_eval.log",
        "outputs/modular-eval/logs/run_alignment_eval.log",
        "outputs/modular-eval/logs/run_diarization_eval.log",
        "outputs/modular-eval/metrics/benchmark_summary.json",
        "outputs/modular-eval/raw/qwen_asr_eval.json",
        "outputs/modular-eval/raw/qwen_aligner_eval.json",
        "outputs/modular-eval/raw/qwen_aligner_f300_chunked.json",
        "outputs/modular-eval/raw/nemotron_diar_eval.json",
        "outputs/modular-eval/raw/nemotron_diar_full_streaming_segments.json",
        "outputs/modular-eval/raw/nemotron_diar_full_offline_segments.json",
        "outputs/modular-eval/raw/nemotron_diar_full_probs_100ms.json",
        "outputs/modular-eval/raw/integration_probe.json",
    ):
        copy_file(ROOT / rel, rel)

    # 4. Bounded French & Mandarin audio fixtures + manifests (exclude >1MB/full media)
    for rel in (
        "outputs/audio/zh/zh_manifest.json",
        "outputs/audio/zh/zh_fleurs_00_1906.wav",
        "outputs/audio/zh/zh_fleurs_01_2006.wav",
        "outputs/audio/zh/zh_fleurs_05_1805.wav",
        "outputs/audio/overlap/fr_fleurs_manifest.json",
        "outputs/audio/overlap/overlap_manifest.json",
        "outputs/audio/overlap/synthetic_zh_2spk_overlap.wav",
        "outputs/audio/overlap/synthetic_fr_2spk_overlap.wav",
        "outputs/audio/fr/fr_0539_0542_overlap_nonlex.wav",
        "outputs/audio/fr/fr_0550_0552_transition.wav",
        "outputs/audio/fr/fr_1031_1033_music_fp.wav",
        "outputs/audio/fr/fr_1731_1735_overlap_vox.wav",
        "outputs/audio/fr/fr_2208_2211_high_pitch.wav",
    ):
        copy_file(ROOT / rel, rel)

    for p in (
        PYANNOTE / "outputs/eval_overlap/baseline_synthetic.json",
        PYANNOTE / "outputs/eval_overlap/synthetic/S1_balanced.wav",
        PYANNOTE / "outputs/eval_overlap/synthetic/S2_quiet_secondary.wav",
        PYANNOTE / "outputs/eval_overlap/synthetic/S3_interrupted_0_9s.wav",
    ):
        copy_file(p, f"fixtures/pyannote_overlap/{p.name}")

    # 5. Git diff & provenance since baseline SHA
    head_sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    pyannote_sha = subprocess.check_output(["git", "-C", str(PYANNOTE), "rev-parse", "HEAD"], text=True).strip()
    git_diff = subprocess.check_output(["git", "-C", str(ROOT), "diff", f"{BASE_SHA}..HEAD"], text=True, errors="replace")
    git_log = subprocess.check_output(["git", "-C", str(ROOT), "log", f"{BASE_SHA}..HEAD", "--stat"], text=True)

    copy_dir = STAGE / "evaluations/modular-bakeoff"
    (copy_dir / "git_provenance.json").write_text(
        json.dumps(
            {
                "base_head": BASE_SHA,
                "current_head": head_sha,
                "pyannote_reference_head": pyannote_sha,
                "source_media_path": "/opt/mcog/Les mystérieuses cités d'or Saison 4 Episode 1 [YlVGUDNbyXk].webm",
                "full_episode_wav_path": "/workspace/nvidia-transcribe/outputs/audio/fr/fr_full_episode.wav",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (copy_dir / "git_diff_since_base.patch").write_text(git_diff, encoding="utf-8")
    (copy_dir / "git_log_since_base.txt").write_text(git_log, encoding="utf-8")

    # 6. Build SHA-256 manifest from exact staged bytes
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
    (STAGE / "evaluations/modular-bakeoff/artifact-manifest.json").write_text(manifest_json, encoding="utf-8")
    MANIFEST_OUT.write_text(manifest_json, encoding="utf-8")

    # 7. Create tar.gz and verify
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with tarfile.open(ARCHIVE, "w:gz") as tf:
        for p in sorted(STAGE.rglob("*")):
            tf.add(p, arcname=str(p.relative_to(STAGE)), recursive=False)

    subprocess.run(["gzip", "-t", str(ARCHIVE)], check=True)
    listed = subprocess.check_output(["tar", "-tzf", str(ARCHIVE)], text=True).splitlines()

    for e in entries:
        staged_p = STAGE / e["path"]
        assert staged_p.stat().st_size == e["bytes"], f"Size mismatch: {e['path']}"
        assert sha256_file(staged_p) == e["sha256"], f"SHA mismatch: {e['path']}"
        assert e["path"] in listed, f"Missing from tarball: {e['path']}"

    for member in listed:
        assert not member.endswith((".safetensors", ".webm", ".mp4", ".nemo", ".pem", ".key")), (
            f"Forbidden weight/media/secret in tarball: {member}"
        )
        assert ".venv/" not in member and ".venv-diar/" not in member and ".hf-cache/" not in member, (
            f"Forbidden venv/cache in tarball: {member}"
        )

    archive_bytes = ARCHIVE.stat().st_size
    archive_sha = sha256_file(ARCHIVE)
    shutil.rmtree(STAGE)

    print(
        json.dumps(
            {
                "archive_path": str(ARCHIVE),
                "bytes": archive_bytes,
                "sha256": archive_sha,
                "verified": True,
                "file_count": len(entries) + 1,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
