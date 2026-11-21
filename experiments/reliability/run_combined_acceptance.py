#!/usr/bin/env python3
"""Combined GPU acceptance suite for Phase A + Phase B repaired pipeline."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path("/workspace/nvidia-transcribe")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(WORKSPACE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from experiments.modular.common import FRENCH_FULL_EPISODE_WEBM, load_shared_corpus
from nvidia_transcribe.client import NemotronVLLMClient
from nvidia_transcribe.metrics import compute_cer, compute_wer
from nvidia_transcribe.recording import RecordingPipelineConfig, process_media

BASE_URL = "http://127.0.0.1:8190/v1"
SERVED_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"
OUT_ROOT = WORKSPACE_ROOT / "outputs/reliability_runs"


def _summarize_contract(contract: dict[str, Any]) -> dict[str, Any]:
    utts = contract.get("utterances") or []
    units = [u for utt in utts for u in (utt.get("source_words") or utt.get("source_characters") or [])]
    timing_statuses: dict[str, int] = {}
    align_statuses: dict[str, int] = {}
    for u in utts:
        ts = (u.get("timestamp_provenance") or {}).get("timing_status", "unknown")
        timing_statuses[ts] = timing_statuses.get(ts, 0) + 1
        ast = str(u.get("alignment_status") or "unknown")
        align_statuses[ast] = align_statuses.get(ast, 0) + 1

    unit_align: dict[str, int] = {}
    for unit in units:
        k = str(unit.get("alignment_status") or "unknown")
        unit_align[k] = unit_align.get(k, 0) + 1

    sr = contract.get("speaker_reconciliation") or {}
    ambs = sr.get("boundary_ambiguities") or []
    unk_preserved = sum(1 for a in ambs if a.get("reason") == "unknown_speaker_cannot_prove_duplicate")
    ext_diags = [d.get("external_aligner") for d in (contract.get("alignment_diagnostics") or []) if d.get("external_aligner")]
    return {
        "schema_version": contract.get("schema_version"),
        "media_duration_s": (contract.get("source") or {}).get("duration_s"),
        "total_chunks": len(sr.get("chunk_summaries") or []),
        "total_utterances": len(utts),
        "total_units": len(units),
        "timed_units": sum(1 for u in units if u.get("valid_timing")),
        "untimed_units": sum(1 for u in units if not u.get("valid_timing")),
        "global_speaker_count": len(contract.get("speakers") or []),
        "utterance_timing_status_counts": timing_statuses,
        "utterance_alignment_status_counts": align_statuses,
        "unit_alignment_status_counts": unit_align,
        "boundary_ambiguity_count": len(ambs),
        "unknown_speaker_non_dedup_count": unk_preserved,
        "skipped_ambiguous_merges_count": len(sr.get("skipped_ambiguous_merges") or []),
        "external_aligner_diagnostics": ext_diags,
        "warning_count": len(contract.get("processing_warnings") or []),
        "warnings": contract.get("processing_warnings") or [],
        "model_provenance": contract.get("model_provenance"),
    }


def _run_case(
    *,
    name: str,
    media_path: str | Path,
    client: NemotronVLLMClient,
    cfg: RecordingPipelineConfig,
    recompute_only: bool = False,
) -> dict[str, Any]:
    out_dir = OUT_ROOT / name
    contract_path = out_dir / "recording.contract.json"
    if recompute_only and contract_path.is_file():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        wall_s = float((contract.get("model_provenance") or {}).get("total_wall_time_s") or 0.0)
    else:
        t0 = time.perf_counter()
        res = process_media(media_path, output_dir=out_dir, client=client, config=cfg)
        wall_s = round(time.perf_counter() - t0, 3)
        contract = res["contract"] if "contract" in res else json.loads(contract_path.read_text(encoding="utf-8"))
    full_text = " ".join(str(u.get("source_text") or "") for u in contract.get("utterances") or []).strip()
    return {
        "name": name,
        "media_path": str(media_path),
        "output_dir": str(out_dir),
        "wall_time_s": wall_s,
        "config": {
            "source_language": cfg.source_language,
            "mode": cfg.mode,
            "chunk_seconds": cfg.chunk_seconds,
            "context_seconds": cfg.context_seconds,
            "alignment_fallback": cfg.alignment_fallback,
            "speaker_embeddings": cfg.speaker_embeddings,
        },
        "summary": _summarize_contract(contract),
        "full_text": full_text,
        "contract_path": str(contract_path),
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    recompute = "--recompute-summary" in sys.argv
    corpus = load_shared_corpus()
    client = NemotronVLLMClient(base_url=BASE_URL, model=SERVED_MODEL, timeout_s=300.0)

    runs: dict[str, Any] = {}

    # 1. Mandarin OOV ('钾') off vs qwen
    zh01 = corpus["mandarin_fleurs"]["zh_fleurs_01_2006"]
    runs["zh_fleurs_01_2006_off"] = _run_case(
        name="zh_fleurs_01_2006_off",
        media_path=zh01["wav_path"],
        client=client,
        cfg=RecordingPipelineConfig(source_language="zh", alignment_fallback="off", resume=False),
        recompute_only=recompute,
    )
    runs["zh_fleurs_01_2006_off"]["cer_vs_ref"] = compute_cer(zh01["raw_transcription"], runs["zh_fleurs_01_2006_off"]["full_text"])

    runs["zh_fleurs_01_2006_qwen"] = _run_case(
        name="zh_fleurs_01_2006_qwen",
        media_path=zh01["wav_path"],
        client=client,
        cfg=RecordingPipelineConfig(source_language="zh", alignment_fallback="qwen", resume=False),
        recompute_only=recompute,
    )
    runs["zh_fleurs_01_2006_qwen"]["cer_vs_ref"] = compute_cer(zh01["raw_transcription"], runs["zh_fleurs_01_2006_qwen"]["full_text"])

    # 2. Mandarin Latin/CJK code-switch spacing check ('Scotturb')
    zh05 = corpus["mandarin_fleurs"]["zh_fleurs_05_1805"]
    runs["zh_fleurs_05_1805_qwen"] = _run_case(
        name="zh_fleurs_05_1805_qwen",
        media_path=zh05["wav_path"],
        client=client,
        cfg=RecordingPipelineConfig(source_language="zh", alignment_fallback="qwen", resume=False),
        recompute_only=recompute,
    )
    runs["zh_fleurs_05_1805_qwen"]["cer_vs_ref"] = compute_cer(zh05["raw_transcription"], runs["zh_fleurs_05_1805_qwen"]["full_text"])

    # 3. Mandarin overlap & multichunk (off vs qwen)
    runs["zh_2spk_overlap_qwen"] = _run_case(
        name="zh_2spk_overlap_qwen",
        media_path=corpus["zh_2spk_overlap"]["wav_path"],
        client=client,
        cfg=RecordingPipelineConfig(source_language="zh", alignment_fallback="qwen", resume=False),
        recompute_only=recompute,
    )
    runs["zh_multichunk_synthetic_off"] = _run_case(
        name="zh_multichunk_synthetic_off",
        media_path=corpus["zh_multichunk_synthetic"]["path"],
        client=client,
        cfg=RecordingPipelineConfig(
            source_language="zh", chunk_seconds=30.0, context_seconds=5.0, min_tail_seconds=8.0, alignment_fallback="off", resume=False
        ),
        recompute_only=recompute,
    )
    runs["zh_multichunk_synthetic_qwen"] = _run_case(
        name="zh_multichunk_synthetic_qwen",
        media_path=corpus["zh_multichunk_synthetic"]["path"],
        client=client,
        cfg=RecordingPipelineConfig(
            source_language="zh", chunk_seconds=30.0, context_seconds=5.0, min_tail_seconds=8.0, alignment_fallback="qwen", resume=False
        ),
        recompute_only=recompute,
    )

    # 4. Focused French excerpts (child voices F1 + 300s recurring)
    runs["fr_child_voices_f1_qwen"] = _run_case(
        name="fr_child_voices_f1_qwen",
        media_path=corpus["french_regions"]["F1"]["path"],
        client=client,
        cfg=RecordingPipelineConfig(source_language="fr", alignment_fallback="qwen", resume=False),
        recompute_only=recompute,
    )
    runs["fr_300s_repaired_qwen"] = _run_case(
        name="fr_300s_repaired_qwen",
        media_path=corpus["french_regions"]["F_300s"]["path"],
        client=client,
        cfg=RecordingPipelineConfig(
            source_language="fr", chunk_seconds=90.0, context_seconds=10.0, min_tail_seconds=15.0, alignment_fallback="qwen", resume=False
        ),
        recompute_only=recompute,
    )

    # 5. Complete fresh 23-min French episode into new output directory
    runs["fr_full_episode_repaired_90s"] = _run_case(
        name="fr_full_episode_repaired_90s",
        media_path=FRENCH_FULL_EPISODE_WEBM,
        client=client,
        cfg=RecordingPipelineConfig(
            source_language="fr", chunk_seconds=90.0, context_seconds=10.0, min_tail_seconds=15.0, alignment_fallback="qwen", resume=False
        ),
        recompute_only=recompute,
    )

    # Baseline comparison against prior chunked-core fr_full_episode_90s
    old_ep_path = WORKSPACE_ROOT / "outputs/chunked_runs/fr_full_episode_90s/recording.contract.json"
    baseline_cmp = {}
    if old_ep_path.is_file():
        old_ep = json.loads(old_ep_path.read_text(encoding="utf-8"))
        baseline_cmp["fr_full_episode_90s_prior"] = _summarize_contract(old_ep)

    out_path = OUT_ROOT / "combined_acceptance_summary.json"
    out_path.write_text(
        json.dumps({"runs": runs, "baseline_comparison": baseline_cmp}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote combined GPU acceptance summary to {out_path}")


if __name__ == "__main__":
    main()
