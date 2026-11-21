#!/usr/bin/env python3
"""Stage, package, and verify the subtitle-generation campaign evidence archive."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path("/workspace/nvidia-transcribe")
BASE_SHA = "3c76089c955513268582c4fefe924c00e098eb12"
STAGE = ROOT / "outputs/handbacks/_staging_subtitles"
ARCHIVE = ROOT / "outputs/handbacks/subtitle-generation-evidence.tar.gz"
MANIFEST_OUT = ROOT / "evaluations/subtitle-generation/artifact-manifest.json"


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


def bounded_contract_excerpt(src_rel: str, dst_rel: str, max_utterances: int = 80) -> None:
    src = ROOT / src_rel
    if not src.is_file():
        return
    data = json.loads(src.read_text(encoding="utf-8"))
    excerpt = {
        "schema_version": data.get("schema_version"),
        "source": data.get("source"),
        "source_language": data.get("source_language"),
        "target_language": data.get("target_language"),
        "refinement": data.get("refinement"),
        "translation": data.get("translation"),
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

    # 1. Core configs, glossary, scripts, and all 8 subtitle-generation reports
    for rel in (
        "AGENTS.md",
        "README.md",
        "pyproject.toml",
        ".gitignore",
        "configs/vllm_rtx5090.env",
        "configs/glossaries/mcog_fr.yaml",
        "scripts/start_vllm_server.sh",
        "evaluations/subtitle-generation/HANDback.md",
        "evaluations/subtitle-generation/architecture.md",
        "evaluations/subtitle-generation/refinement-evaluation.md",
        "evaluations/subtitle-generation/translation-comparison.md",
        "evaluations/subtitle-generation/subtitle-quality-review.md",
        "evaluations/subtitle-generation/render-validation.md",
        "evaluations/subtitle-generation/test-results.md",
        "evaluations/subtitle-generation/benchmark.json",
    ):
        copy_item(rel)

    # 2. All production source, subtitle experiment scripts, and CPU unit tests
    for folder in ("src/nvidia_transcribe", "tests", "experiments/subtitle", "experiments/modular/tests"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                copy_item(str(path.relative_to(ROOT)))

    # 3. Complete French episode English SRT/ASS + text layers + metadata + profile/translation benchmarks
    fr_ep_dir = ROOT / "outputs/subtitle_runs/fr_full_episode"
    for p in sorted(fr_ep_dir.iterdir()):
        if p.is_file() and p.suffix in (".srt", ".ass", ".txt") or p.name == "subtitles.metadata.json":
            copy_item(str(p.relative_to(ROOT)))
    bounded_contract_excerpt(
        "outputs/subtitle_runs/fr_full_episode/translated.contract.json",
        "outputs/subtitle_runs/fr_full_episode/translated.contract.excerpt.json",
        max_utterances=80,
    )

    for rel in (
        "outputs/subtitle_runs/translation_comparison.json",
        "outputs/subtitle_runs/profile_comparison/profile_comparison.json",
        "outputs/subtitle_runs/zh_qwen_fix_verify/qwen_fix_summary.json",
        "outputs/subtitle_runs/zh_samples/mandarin_subtitles_summary.json",
    ):
        if (ROOT / rel).is_file():
            copy_item(rel)

    # 4. Complete Mandarin subtitle outputs & repaired contracts (compact!)
    zh_samples_dir = ROOT / "outputs/subtitle_runs/zh_samples"
    for p in sorted(zh_samples_dir.rglob("*")):
        if p.is_file() and not p.name.endswith(".wav"):
            copy_item(str(p.relative_to(ROOT)))

    # 5. Visual verification PNG screenshots + manifest
    shots_dir = ROOT / "outputs/subtitle_runs/visual_screenshots"
    for p in sorted(shots_dir.rglob("*")):
        if p.is_file() and p.suffix in (".png", ".json"):
            copy_item(str(p.relative_to(ROOT)))

    # 6. Git provenance & diff since campaign baseline HEAD
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
    (STAGE / "evaluations/subtitle-generation/git_provenance.json").write_text(
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
    (STAGE / "evaluations/subtitle-generation/git_diff_since_base.patch").write_text(git_diff, encoding="utf-8")
    (STAGE / "evaluations/subtitle-generation/git_log_since_base.txt").write_text(git_log, encoding="utf-8")

    # 7. SHA-256 manifest
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
    (STAGE / "evaluations/subtitle-generation/artifact-manifest.json").write_text(manifest_json, encoding="utf-8")

    # 8. Build tar.gz and verify
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
