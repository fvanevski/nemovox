#!/usr/bin/env python3
"""Stage, package, and verify the Mandarin EP166 baseline evidence archive."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path("/workspace/nvidia-transcribe")
BASE_SHA = "e63a68b2ac45a120c36d79967279eae17faf33e4"
STAGE = ROOT / "outputs/handbacks/_staging_ep166"
ARCHIVE = ROOT / "outputs/handbacks/mandarin-ep166-evidence.tar.gz"
MANIFEST_OUT = ROOT / "evaluations/mandarin-ep166/artifact-manifest.json"
RUN_DIR = ROOT / "outputs/subtitle_runs/zh_full_episode_ep166"


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


def bounded_tail_copy(src_rel: str, dst_rel: str, max_lines: int = 300) -> None:
    src = ROOT / src_rel
    if not src.is_file():
        return
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) > max_lines:
        content = "\n".join(lines[:40] + ["", f"... ({len(lines) - max_lines} middle lines omitted) ...", ""] + lines[-(max_lines - 40) :]) + "\n"
    else:
        content = "\n".join(lines) + "\n"
    dst = STAGE / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(content, encoding="utf-8")


def main() -> int:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True, exist_ok=True)
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)

    # 1. Configs, glossary, scripts, and all EP166 reports
    for rel in (
        "AGENTS.md",
        "configs/vllm_rtx5090.env",
        "configs/glossaries/a_will_eternal_zh.yaml",
        "scripts/start_vllm_server.sh",
        "experiments/subtitle/analyze_ep166_baseline.py",
        "experiments/subtitle/package_mandarin_ep166_evidence.py",
        "evaluations/mandarin-ep166/HANDback.md",
        "evaluations/mandarin-ep166/source-media.md",
        "evaluations/mandarin-ep166/run-configuration.md",
        "evaluations/mandarin-ep166/quality-review.md",
        "evaluations/mandarin-ep166/french-mandarin-comparison.md",
        "evaluations/mandarin-ep166/timing-analysis.json",
        "evaluations/mandarin-ep166/benchmark.json",
    ):
        copy_item(rel)

    # 2. Complete EP166 contracts, transcripts, subtitles, metadata, and review diagnostics/screenshots
    for rel in (
        "outputs/subtitle_runs/zh_full_episode_ep166/asr/manifest.json",
        "outputs/subtitle_runs/zh_full_episode_ep166/asr/transcript.txt",
        "outputs/subtitle_runs/zh_full_episode_ep166/asr/recording.contract.json",
        "outputs/subtitle_runs/zh_full_episode_ep166/refined.contract.json",
        "outputs/subtitle_runs/zh_full_episode_ep166/translated.contract.json",
        "outputs/subtitle_runs/zh_full_episode_ep166/raw_source_transcript.txt",
        "outputs/subtitle_runs/zh_full_episode_ep166/refined_source_transcript.txt",
        "outputs/subtitle_runs/zh_full_episode_ep166/final_english_subtitles.txt",
        "outputs/subtitle_runs/zh_full_episode_ep166/subtitles.metadata.json",
        "outputs/subtitle_runs/zh_full_episode_ep166/A Will Eternal Final Season EP166 [e4g3i65Kyf8].en.srt",
        "outputs/subtitle_runs/zh_full_episode_ep166/A Will Eternal Final Season EP166 [e4g3i65Kyf8].en.ass",
        "outputs/subtitle_runs/zh_full_episode_ep166/review/source-metadata.json",
        "outputs/subtitle_runs/zh_full_episode_ep166/review/quality-summary.json",
        "outputs/subtitle_runs/zh_full_episode_ep166/review/timing-analysis.json",
    ):
        copy_item(rel)

    for p in sorted((RUN_DIR / "review/screenshots").glob("*.png")):
        copy_item(str(p.relative_to(ROOT)))

    # 3. Bounded model/runtime logs (no giant stderr tqdm dumps)
    for log_rel in (
        "outputs/logs/vllm_ep166.log",
        "outputs/logs/asr_ep166_off.log",
        "outputs/logs/subtitles_ep166.log",
        "outputs/logs/ninfer_ep166.log",
    ):
        bounded_tail_copy(log_rel, f"evaluations/mandarin-ep166/logs/{Path(log_rel).name}", max_lines=250)

    # 4. CPU test summary + Git provenance & diff since baseline HEAD
    test_out = subprocess.check_output(
        [str(ROOT / ".venv/bin/pytest"), "-v", "-m", "not gpu"],
        cwd=str(ROOT),
        text=True,
    )
    (STAGE / "evaluations/mandarin-ep166/cpu_pytest.txt").write_text(test_out, encoding="utf-8")

    head_sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    pyannote_sha = subprocess.check_output(["git", "-C", "/workspace/pyannote", "rev-parse", "HEAD"], text=True).strip()
    git_diff = subprocess.check_output(["git", "-C", str(ROOT), "diff", f"{BASE_SHA}..HEAD"], text=True, errors="replace")
    git_log = subprocess.check_output(["git", "-C", str(ROOT), "log", f"{BASE_SHA}..HEAD", "--stat"], text=True)

    (STAGE / "evaluations/mandarin-ep166/git_provenance.json").write_text(
        json.dumps({"starting_head": BASE_SHA, "current_head": head_sha, "pyannote_head": pyannote_sha}, indent=2),
        encoding="utf-8",
    )
    (STAGE / "evaluations/mandarin-ep166/git_diff_since_base.patch").write_text(git_diff, encoding="utf-8")
    (STAGE / "evaluations/mandarin-ep166/git_log_since_base.txt").write_text(git_log, encoding="utf-8")

    # 5. Build SHA-256 manifest
    entries = []
    for p in sorted(STAGE.rglob("*")):
        if p.is_file():
            entries.append({"path": str(p.relative_to(STAGE)), "bytes": p.stat().st_size, "sha256": sha256_file(p)})

    manifest = {
        "archive": str(ARCHIVE),
        "starting_head": BASE_SHA,
        "current_head": head_sha,
        "file_count": len(entries),
        "total_staged_bytes": sum(e["bytes"] for e in entries),
        "files": entries,
    }
    manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2)
    MANIFEST_OUT.write_text(manifest_json, encoding="utf-8")
    (STAGE / "evaluations/mandarin-ep166/artifact-manifest.json").write_text(manifest_json, encoding="utf-8")

    # 6. Create & verify archive
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with tarfile.open(ARCHIVE, "w:gz") as tf:
        for p in sorted(STAGE.rglob("*")):
            tf.add(p, arcname=str(p.relative_to(STAGE)), recursive=False)

    subprocess.run(["gzip", "-t", str(ARCHIVE)], check=True)
    listed = set(subprocess.check_output(["tar", "-tzf", str(ARCHIVE)], text=True).splitlines())
    for e in entries:
        assert e["path"] in listed, f"Missing in tarball: {e['path']}"
        assert sha256_file(STAGE / e["path"]) == e["sha256"], f"SHA mismatch: {e['path']}"

    shutil.rmtree(STAGE)
    print(f"Verified archive {ARCHIVE} ({ARCHIVE.stat().st_size} bytes, {len(entries)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
