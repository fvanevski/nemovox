#!/usr/bin/env python3
"""Stage, package, and verify the Viewing-Led Subtitle Validation evidence archive."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path("/workspace/nvidia-transcribe")
BASE_SHA = "c93962bd6897727326d69b625ce761f59801e242"
STAGE = ROOT / "outputs/handbacks/_staging_viewing_validation"
ARCHIVE = ROOT / "outputs/handbacks/viewing-led-validation-evidence.tar.gz"
EVAL_DIR = ROOT / "evaluations/viewing-led-validation"
MANIFEST_OUT = EVAL_DIR / "artifact-manifest.json"


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

    # 1. Reports, overlays, scripts, source code, and CPU regression tests
    for rel in (
        "AGENTS.md",
        "configs/overlays/fr_s04e01_validation.json",
        "configs/overlays/zh_ep166_validation.json",
        "experiments/subtitle/analyze_ep166_baseline.py",
        "experiments/subtitle/package_viewing_validation_evidence.py",
        "evaluations/viewing-led-validation/HANDback.md",
        "evaluations/viewing-led-validation/scene-review.md",
        "evaluations/viewing-led-validation/mandarin-untimed-triage.md",
        "evaluations/viewing-led-validation/translation-quality-review.md",
        "evaluations/viewing-led-validation/presentation-review.md",
        "evaluations/viewing-led-validation/before-after-summary.md",
        "evaluations/viewing-led-validation/test-results.md",
        "evaluations/viewing-led-validation/viewing-playlist.md",
        "evaluations/viewing-led-validation/review-scenes.json",
        "evaluations/viewing-led-validation/untimed-decisions.json",
        "evaluations/viewing-led-validation/comparison_metrics.json",
        "evaluations/viewing-led-validation/preview_clips_manifest.json",
    ):
        copy_item(rel)

    for p in sorted((ROOT / "src/nvidia_transcribe").glob("*.py")):
        copy_item(str(p.relative_to(ROOT)))
    for p in sorted((ROOT / "tests").glob("test_*.py")):
        copy_item(str(p.relative_to(ROOT)))

    # 2. Pre-validation vs Post-validation subtitle tracks & contracts
    fr_stem = "Les mystérieuses cités d'or Saison 4 Episode 1 [YlVGUDNbyXk]"
    zh_stem = "A Will Eternal Final Season EP166 [e4g3i65Kyf8]"
    for prefix, stem in (
        ("outputs/subtitle_runs/viewing_quality/fr_full_episode", fr_stem),
        ("outputs/subtitle_runs/viewing_validation/fr_full_episode", fr_stem),
        ("outputs/subtitle_runs/viewing_quality/zh_full_episode_ep166", zh_stem),
        ("outputs/subtitle_runs/viewing_validation/zh_full_episode_ep166", zh_stem),
    ):
        for fname in (
            f"{stem}.en.srt",
            f"{stem}.en.ass",
            "final_english_subtitles.txt",
            "subtitles.metadata.json",
            "translated.contract.json",
        ):
            rel = f"{prefix}/{fname}"
            if (ROOT / rel).is_file():
                copy_item(rel)

    # 3. Rendered 1080p verification PNG screenshots & bounded key playable excerpts
    for p in sorted((ROOT / "outputs/subtitle_runs/viewing_validation/screenshots").glob("*.png")):
        copy_item(str(p.relative_to(ROOT)))

    for clip_name in (
        "fr_04_0500_0522_3spk_topalt.mp4",
        "zh_06_1028_1058_chunk7_missing.mp4",
        "zh_08_1620_1655_chunk10_11_monologue.mp4",
        "zh_09_1740_1815_chunk11_rebuke.mp4",
    ):
        rel = f"outputs/subtitle_runs/viewing_validation/previews/{clip_name}"
        if (ROOT / rel).is_file():
            copy_item(rel)

    # 4. CPU pytest output + Git provenance/diff since starting HEAD
    test_out = subprocess.check_output(
        [str(ROOT / ".venv/bin/pytest"), "-v", "-m", "not gpu"],
        cwd=str(ROOT),
        text=True,
    )
    (STAGE / "evaluations/viewing-led-validation/cpu_pytest.txt").write_text(test_out, encoding="utf-8")

    head_sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    pyannote_sha = subprocess.check_output(["git", "-C", "/workspace/pyannote", "rev-parse", "HEAD"], text=True).strip()
    git_diff = subprocess.check_output(["git", "-C", str(ROOT), "diff", f"{BASE_SHA}..HEAD"], text=True, errors="replace")
    git_log = subprocess.check_output(["git", "-C", str(ROOT), "log", f"{BASE_SHA}..HEAD", "--stat"], text=True)

    (STAGE / "evaluations/viewing-led-validation/git_provenance.json").write_text(
        json.dumps({"starting_head": BASE_SHA, "current_head": head_sha, "pyannote_head": pyannote_sha}, indent=2) + "\n",
        encoding="utf-8",
    )
    (STAGE / "evaluations/viewing-led-validation/git_diff_since_base.patch").write_text(git_diff, encoding="utf-8")
    (STAGE / "evaluations/viewing-led-validation/git_log_since_base.txt").write_text(git_log, encoding="utf-8")

    # 5. Build SHA-256 manifest
    entries = []
    for p in sorted(STAGE.rglob("*")):
        if p.is_file():
            entries.append({"path": str(p.relative_to(STAGE)), "bytes": p.stat().st_size, "sha256": sha256_file(p)})

    manifest = {
        "archive": str(ARCHIVE),
        "starting_head": BASE_SHA,
        "current_head": head_sha,
        "pyannote_head": pyannote_sha,
        "file_count": len(entries),
        "total_staged_bytes": sum(e["bytes"] for e in entries),
        "files": entries,
    }
    manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    MANIFEST_OUT.write_text(manifest_json, encoding="utf-8")
    (STAGE / "evaluations/viewing-led-validation/artifact-manifest.json").write_text(manifest_json, encoding="utf-8")

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
