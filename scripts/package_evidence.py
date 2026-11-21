#!/usr/bin/env python3
"""Stage and verify the standalone evaluation evidence archive."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path("/workspace/nvidia-transcribe")
STAGE = ROOT / "outputs/handbacks/_staging"
ARCHIVE = ROOT / "outputs/handbacks/nemotron35-nvfp4-evaluation.tar.gz"
MANIFEST_OUT = ROOT / "evaluations/nemotron35/artifact-manifest.json"


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


def main() -> int:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True, exist_ok=True)
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)

    # 1. Core project files & reports
    for rel in (
        "AGENTS.md",
        "README.md",
        "pyproject.toml",
        ".gitignore",
        "configs/vllm_rtx5090.env",
        "scripts/start_vllm_server.sh",
        "scripts/run_evaluation.py",
        "scripts/test_pyannote_subtitle_compat.py",
        "scripts/package_evidence.py",
        "evaluations/nemotron35/HANDback.md",
        "evaluations/nemotron35/environment.md",
        "evaluations/nemotron35/benchmark.json",
        "evaluations/nemotron35/quality-review.md",
        "evaluations/nemotron35/runtime-contract.md",
        "evaluations/nemotron35/test-results.md",
        "evaluations/nemotron35/architecture-assessment.md",
        "evaluations/nemotron35/checkpoint_manifest.json",
        "evaluations/nemotron35/pip_list.txt",
        "evaluations/nemotron35/pytest_cpu.txt",
    ):
        copy_item(rel)

    # 2. Source & CPU tests
    for folder in ("src/nvidia_transcribe", "tests"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                copy_item(str(path.relative_to(ROOT)))

    # 3. Representative small responses & rendered subtitles
    sample_responses = (
        "outputs/responses/smoke_fr_0550.contract.json",
        "outputs/responses/smoke_fr_0550.raw.json",
        "outputs/responses/fr/fr_0316_0355_child_voices_speaker.contract.json",
        "outputs/responses/fr/fr_0305_0445_cont_100s_speaker.contract.json",
        "outputs/responses/fr/fr_0539_0542_overlap_nonlex_rich.contract.json",
        "outputs/responses/fr/fr_2208_2211_high_pitch_rich.contract.json",
        "outputs/responses/zh/zh_fleurs_00_1906_auto.contract.json",
        "outputs/responses/zh/zh_fleurs_01_2006_auto.contract.json",
        "outputs/responses/zh/zh_fleurs_01_2006_auto.raw.json",
        "outputs/responses/zh/zh_fleurs_05_1805_explicit.contract.json",
        "outputs/responses/overlap/fr_2spk_overlap_sot.contract.json",
        "outputs/responses/overlap/zh_2spk_overlap_sot.contract.json",
        "outputs/responses/overlap/zh_2spk_seq_speaker.contract.json",
        "outputs/renders/fr_0316_0355_nemotron.srt",
        "outputs/renders/fr_0316_0355_nemotron.ass",
        "outputs/renders/zh_sample_nemotron.srt",
        "outputs/renders/zh_sample_nemotron.ass",
        "outputs/renders/compat_report.json",
    )
    for rel in sample_responses:
        if (ROOT / rel).is_file():
            copy_item(rel)

    # 4. Bounded representative audio excerpts (exclude 100s/300s/full-episode WAVs)
    bounded_audio = (
        "outputs/audio/fr/fr_0539_0542_overlap_nonlex.wav",
        "outputs/audio/fr/fr_0550_0552_transition.wav",
        "outputs/audio/fr/fr_1031_1033_music_fp.wav",
        "outputs/audio/fr/fr_1731_1735_overlap_vox.wav",
        "outputs/audio/fr/fr_2208_2211_high_pitch.wav",
        "outputs/audio/zh/zh_manifest.json",
        "outputs/audio/zh/zh_fleurs_00_1906.wav",
        "outputs/audio/zh/zh_fleurs_01_2006.wav",
        "outputs/audio/overlap/overlap_manifest.json",
        "outputs/audio/overlap/fr_fleurs_manifest.json",
        "outputs/audio/overlap/synthetic_fr_2spk_overlap.wav",
        "outputs/audio/overlap/synthetic_zh_2spk_overlap.wav",
    )
    for rel in bounded_audio:
        if (ROOT / rel).is_file():
            copy_item(rel)

    # 5. Bounded vLLM server log excerpt + Git state
    log_path = ROOT / "outputs/logs/vllm_server.log"
    if log_path.is_file():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        excerpt = "\n".join(lines[:140] + ["\n... [TRUNCATED MIDDLE] ...\n"] + lines[-120:]) + "\n"
        dst_log = STAGE / "outputs/logs/vllm_server_excerpt.log"
        dst_log.parent.mkdir(parents=True, exist_ok=True)
        dst_log.write_text(excerpt, encoding="utf-8")

    git_sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    nemo_sha = subprocess.check_output(
        ["git", "-C", str(ROOT / "vendor/nemo-speech"), "rev-parse", "HEAD"], text=True
    ).strip()
    pyannote_sha = subprocess.check_output(
        ["git", "-C", "/workspace/pyannote", "rev-parse", "HEAD"], text=True
    ).strip()
    git_log = subprocess.check_output(["git", "-C", str(ROOT), "log", "-n", "5", "--stat"], text=True)
    (STAGE / "evaluations/nemotron35/git_provenance.json").write_text(
        json.dumps(
            {
                "nvidia_transcribe_head": git_sha,
                "nemo_speech_head": nemo_sha,
                "pyannote_reference_head": pyannote_sha,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (STAGE / "evaluations/nemotron35/git_log.txt").write_text(git_log, encoding="utf-8")

    # 6. Build checksum manifest from staged files
    entries = []
    for p in sorted(STAGE.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(STAGE))
            entries.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha256_file(p)})

    manifest = {
        "archive": str(ARCHIVE),
        "nvidia_transcribe_head": git_sha,
        "nemo_speech_head": nemo_sha,
        "file_count": len(entries),
        "total_staged_bytes": sum(e["bytes"] for e in entries),
        "files": entries,
    }
    manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2)
    (STAGE / "evaluations/nemotron35/artifact-manifest.json").write_text(manifest_json, encoding="utf-8")
    MANIFEST_OUT.write_text(manifest_json, encoding="utf-8")

    # 7. Create tar.gz and verify
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with tarfile.open(ARCHIVE, "w:gz") as tf:
        for p in sorted(STAGE.rglob("*")):
            tf.add(p, arcname=str(p.relative_to(STAGE)), recursive=False)

    subprocess.run(["gzip", "-t", str(ARCHIVE)], check=True)
    listed = subprocess.check_output(["tar", "-tzf", str(ARCHIVE)], text=True).splitlines()

    # Verify every staged file matches manifest size & sha256
    for e in entries:
        staged_p = STAGE / e["path"]
        assert staged_p.stat().st_size == e["bytes"], f"Size mismatch: {e['path']}"
        assert sha256_file(staged_p) == e["sha256"], f"SHA mismatch: {e['path']}"
        assert e["path"] in listed, f"Missing from tarball: {e['path']}"

    # Verify no forbidden files inside tarball
    for member in listed:
        assert not member.endswith((".safetensors", ".webm", ".mp4", ".pem", ".key")), f"Forbidden file: {member}"
        assert ".venv/" not in member and "vendor/nemo-speech/" not in member, f"Forbidden dir: {member}"

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
