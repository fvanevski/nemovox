#!/usr/bin/env python3
"""Stage, package, and verify the Dual-Language Viewing-Quality Remediation evidence archive."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path("/workspace/nvidia-transcribe")
BASE_SHA = "e8cbc7c83dc88dd5dca3fe96c0e06d52e2081564"
STAGE = ROOT / "outputs/handbacks/_staging_viewing_quality"
ARCHIVE = ROOT / "outputs/handbacks/viewing-quality-evidence.tar.gz"
EVAL_DIR = ROOT / "evaluations/viewing-quality"
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


def bounded_tail_copy(src_rel: str, dst_rel: str, max_lines: int = 300) -> None:
    src = ROOT / src_rel
    if not src.is_file():
        return
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) > max_lines:
        content = "\n".join(
            lines[:40]
            + ["", f"... ({len(lines) - max_lines} middle lines omitted) ...", ""]
            + lines[-(max_lines - 40) :]
        ) + "\n"
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

    # 1. Core docs, evaluation reports, scripts, source code, and tests
    for rel in (
        "AGENTS.md",
        "configs/glossaries/a_will_eternal_zh.yaml",
        "experiments/subtitle/analyze_ep166_baseline.py",
        "experiments/subtitle/run_viewing_quality.py",
        "experiments/subtitle/package_viewing_quality_evidence.py",
        "evaluations/viewing-quality/HANDback.md",
        "evaluations/viewing-quality/implementation-review.md",
        "evaluations/viewing-quality/untimed-dialogue-audit.md",
        "evaluations/viewing-quality/cache-regressions.md",
        "evaluations/viewing-quality/readability-comparison.md",
        "evaluations/viewing-quality/visual-review.md",
        "evaluations/viewing-quality/test-results.md",
        "evaluations/viewing-quality/comparison_metrics.json",
        "evaluations/viewing-quality/benchmark.json",
    ):
        copy_item(rel)

    for p in sorted((ROOT / "src/nvidia_transcribe").glob("*.py")):
        copy_item(str(p.relative_to(ROOT)))
    for p in sorted((ROOT / "tests").glob("test_*.py")):
        copy_item(str(p.relative_to(ROOT)))

    # 2. Baseline vs regenerated subtitle tracks & contracts
    fr_stem = "Les mystérieuses cités d'or Saison 4 Episode 1 [YlVGUDNbyXk]"
    zh_stem = "A Will Eternal Final Season EP166 [e4g3i65Kyf8]"
    for base_prefix, stem in (
        ("outputs/subtitle_runs/fr_full_episode", fr_stem),
        ("outputs/subtitle_runs/viewing_quality/fr_full_episode", fr_stem),
        ("outputs/subtitle_runs/zh_full_episode_ep166", zh_stem),
        ("outputs/subtitle_runs/viewing_quality/zh_full_episode_ep166", zh_stem),
    ):
        for fname in (
            f"{stem}.en.srt",
            f"{stem}.en.ass",
            "raw_source_transcript.txt",
            "refined_source_transcript.txt",
            "final_english_subtitles.txt",
            "subtitles.metadata.json",
            "translated.contract.json",
        ):
            rel = f"{base_prefix}/{fname}"
            if (ROOT / rel).is_file():
                copy_item(rel)

    for shot_dir in (
        ROOT / "outputs/subtitle_runs/viewing_quality/fr_full_episode/screenshots",
        ROOT / "outputs/subtitle_runs/viewing_quality/zh_full_episode_ep166/screenshots",
    ):
        for p in sorted(shot_dir.glob("*.png")):
            copy_item(str(p.relative_to(ROOT)))

    # 3. Extract dedicated untimed audit JSON artifact for compact review
    zh_contract = json.loads(
        (ROOT / "outputs/subtitle_runs/viewing_quality/zh_full_episode_ep166/translated.contract.json").read_text(
            encoding="utf-8"
        )
    )
    (STAGE / "evaluations/viewing-quality/ep166_untimed_audit.json").write_text(
        json.dumps(zh_contract.get("untimed_audit") or [], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # 4. Bounded logs
    for log_rel in ("outputs/logs/ninfer_vq.log", "outputs/logs/vq_pipeline.log"):
        bounded_tail_copy(log_rel, f"evaluations/viewing-quality/logs/{Path(log_rel).name}", max_lines=250)

    # 5. CPU pytest output + Git provenance/diff
    test_out = subprocess.check_output(
        [str(ROOT / ".venv/bin/pytest"), "-v", "-m", "not gpu"],
        cwd=str(ROOT),
        text=True,
    )
    (STAGE / "evaluations/viewing-quality/cpu_pytest.txt").write_text(test_out, encoding="utf-8")

    head_sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    pyannote_sha = subprocess.check_output(["git", "-C", "/workspace/pyannote", "rev-parse", "HEAD"], text=True).strip()
    git_diff = subprocess.check_output(["git", "-C", str(ROOT), "diff", f"{BASE_SHA}..HEAD"], text=True, errors="replace")
    git_log = subprocess.check_output(["git", "-C", str(ROOT), "log", f"{BASE_SHA}..HEAD", "--stat"], text=True)

    (STAGE / "evaluations/viewing-quality/git_provenance.json").write_text(
        json.dumps({"starting_head": BASE_SHA, "current_head": head_sha, "pyannote_head": pyannote_sha}, indent=2) + "\n",
        encoding="utf-8",
    )
    (STAGE / "evaluations/viewing-quality/git_diff_since_base.patch").write_text(git_diff, encoding="utf-8")
    (STAGE / "evaluations/viewing-quality/git_log_since_base.txt").write_text(git_log, encoding="utf-8")

    # 6. SHA-256 manifest
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
    (STAGE / "evaluations/viewing-quality/artifact-manifest.json").write_text(manifest_json, encoding="utf-8")

    # 7. Create & verify tarball
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
