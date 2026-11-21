#!/usr/bin/env python3
"""Stage and verify the chunked-core implementation evidence archive."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path("/workspace/nvidia-transcribe")
BASE_SHA = "0fc70b1d5d15da616458f5f11915ecd4c31959af"
STAGE = ROOT / "outputs/handbacks/_staging_chunked"
ARCHIVE = ROOT / "outputs/handbacks/chunked-core-evidence.tar.gz"
MANIFEST_OUT = ROOT / "evaluations/chunked-core/artifact-manifest.json"


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


def bounded_contract_excerpt(src_rel: str, dst_rel: str, max_utterances: int = 35, max_chunks: int = 6) -> None:
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
        "processing": {
            "chunk_config": data.get("processing", {}).get("chunk_config"),
            "chunk_count": data.get("processing", {}).get("chunk_count"),
            "boundary_reconciliation": data.get("processing", {}).get("boundary_reconciliation"),
            "chunks_excerpt": [
                {
                    "chunk_index": c.get("chunk_index"),
                    "core_start_s": c.get("core_start_s"),
                    "core_end_s": c.get("core_end_s"),
                    "alignment_status": c.get("alignment_status"),
                    "local_speakers": c.get("local_speakers"),
                    "speaker_mapping": c.get("speaker_mapping"),
                    "diarization_count": len(c.get("diarization") or []),
                }
                for c in (data.get("processing", {}).get("chunks") or [])[:max_chunks]
            ],
        },
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

    # 1. Core docs, configs, and all chunked-core reports
    for rel in (
        "AGENTS.md",
        "README.md",
        "pyproject.toml",
        ".gitignore",
        "configs/vllm_rtx5090.env",
        "scripts/start_vllm_server.sh",
        "scripts/run_chunked_evaluation.py",
        "scripts/package_chunked_evidence.py",
        "evaluations/chunked-core/HANDback.md",
        "evaluations/chunked-core/architecture.md",
        "evaluations/chunked-core/contract.md",
        "evaluations/chunked-core/benchmark.json",
        "evaluations/chunked-core/supplemental.json",
        "evaluations/chunked-core/quality-review.md",
        "evaluations/chunked-core/test-results.md",
        "evaluations/chunked-core/speaker-reconciliation.md",
        "evaluations/chunked-core/alignment-diagnostics.md",
        "evaluations/chunked-core/pytest_cpu.txt",
    ):
        copy_item(rel)

    # 2. All production source and CPU tests
    for folder in ("src/nvidia_transcribe", "tests"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                copy_item(str(path.relative_to(ROOT)))

    # 3. Bounded representative recording artifacts (never full audio or huge raw dumps)
    for rel in (
        "outputs/chunked_runs/fr_full_episode_90s/transcript.txt",
        "outputs/chunked_runs/cli_smoke_fr39s/transcript.txt",
        "outputs/chunked_runs/cli_smoke_fr39s/recording.contract.json",
        "outputs/chunked_runs/zh_fleurs_01_2006/recording.contract.json",
        "outputs/chunked_runs/zh_fleurs_01_2006/chunks/chunk_000.raw.json",
        "outputs/chunked_runs/zh_2spk_overlap/recording.contract.json",
        "outputs/chunked_runs/zh_2spk_overlap/chunks/chunk_000.raw.json",
        "outputs/chunked_runs/zh_multichunk_synthetic/recording.contract.json",
        "outputs/chunked_runs/zh_multichunk_synthetic/transcript.txt",
    ):
        if (ROOT / rel).is_file():
            copy_item(rel)

    bounded_contract_excerpt(
        "outputs/chunked_runs/fr_full_episode_90s/recording.contract.json",
        "outputs/samples/fr_full_episode_90s.contract.excerpt.json",
        max_utterances=60,
        max_chunks=16,
    )
    bounded_contract_excerpt(
        "outputs/chunked_runs/fr_full_episode_90s_temporal_only/recording.contract.json",
        "outputs/samples/fr_full_episode_90s_temporal_only.contract.excerpt.json",
        max_utterances=20,
        max_chunks=16,
    )

    # 4. Git diff & provenance against baseline HEAD
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
    (STAGE / "evaluations/chunked-core/git_provenance.json").write_text(
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
    (STAGE / "evaluations/chunked-core/git_diff_since_base.patch").write_text(git_diff, encoding="utf-8")
    (STAGE / "evaluations/chunked-core/git_log_since_base.txt").write_text(git_log, encoding="utf-8")

    # 5. Build SHA-256 manifest over staged files
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
    (STAGE / "evaluations/chunked-core/artifact-manifest.json").write_text(manifest_json, encoding="utf-8")
    MANIFEST_OUT.write_text(manifest_json, encoding="utf-8")

    # 6. Create tar.gz and verify
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
        assert not member.endswith((".safetensors", ".webm", ".mp4", ".wav", ".pem", ".key")), (
            f"Forbidden media/weight/secret in tarball: {member}"
        )
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
