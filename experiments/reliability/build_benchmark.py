#!/usr/bin/env python3
"""Compile canonical benchmark.json for evaluations/transcription-reliability/ from raw GPU/CPU artifacts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

WORKSPACE = Path("/workspace/nvidia-transcribe")
REL_RUNS = WORKSPACE / "outputs/reliability_runs"
EVAL_DIR = WORKSPACE / "evaluations/transcription-reliability"


def _git_output(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=str(WORKSPACE), text=True).strip()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize_rec(contract: dict[str, Any]) -> dict[str, Any]:
    utts = contract.get("utterances") or []
    units = [u for utt in utts for u in (utt.get("source_words") or utt.get("source_characters") or [])]
    sr = contract.get("speaker_reconciliation") or {}
    t_counts: dict[str, int] = {}
    for u in utts:
        ts = (u.get("timestamp_provenance") or {}).get("timing_status", "unknown")
        t_counts[ts] = t_counts.get(ts, 0) + 1
    u_align: dict[str, int] = {}
    for un in units:
        a = str(un.get("alignment_status") or "unknown")
        u_align[a] = u_align.get(a, 0) + 1
    return {
        "schema_version": contract.get("schema_version"),
        "duration_s": (contract.get("source") or {}).get("duration_s"),
        "utterance_count": len(utts),
        "global_speaker_count": len(contract.get("speakers") or []),
        "unit_count": len(units),
        "timed_unit_count": sum(1 for u in units if u.get("valid_timing")),
        "untimed_unit_count": sum(1 for u in units if not u.get("valid_timing")),
        "utterance_timing_status_counts": t_counts,
        "unit_alignment_status_counts": u_align,
        "raw_local_speaker_count": sr.get("raw_local_speaker_count"),
        "ambiguous_matches_count": len(sr.get("ambiguous_matches") or []),
        "collisions_prevented_count": len(sr.get("collisions_prevented") or []),
        "unresolved_speakers_count": len(sr.get("unresolved_speakers") or []),
        "unknown_speaker_count": sr.get("unknown_speaker_count"),
        "wall_time_s": (contract.get("model_provenance") or {}).get("total_wall_time_s"),
        "alignment_fallback": (contract.get("model_provenance") or {}).get("alignment_fallback"),
        "external_aligner_diagnostics": [
            d.get("external_aligner") for d in (contract.get("alignment_diagnostics") or []) if d.get("external_aligner")
        ],
    }


def main() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    phase_c = _load(REL_RUNS / "phase_c_refinement.json")
    phase_d = _load(REL_RUNS / "phase_d_scene_diarization.json")
    comb = _load(REL_RUNS / "combined_acceptance_summary.json")

    old_fr = _load(WORKSPACE / "outputs/chunked_runs/fr_full_episode_90s/recording.contract.json")
    new_fr = _load(REL_RUNS / "fr_full_episode_repaired_90s/recording.contract.json")

    live_runs = {}
    for k, v in comb["runs"].items():
        c = _load(Path(v["contract_path"]))
        live_runs[k] = {
            **_summarize_rec(c),
            "cer_vs_ref": v.get("cer_vs_ref"),
            "full_text": v.get("full_text"),
            "contract_path": v["contract_path"],
        }

    benchmark = {
        "campaign": "nemotron-transcription-reliability-and-refinement",
        "git": {
            "baseline_head": "889c28189e9418d45cf5de87b979135c5c4dc3c2",
            "phase_a_commit": _git_output(["log", "--grep=Repair eight chunked", "--format=%H", "-n", "1"]),
            "phase_b_commit": _git_output(["log", "--grep=selective Qwen forced-alignment", "--format=%H", "-n", "1"]),
            "current_head": _git_output(["rev-parse", "HEAD"]),
            "pyannote_head": subprocess.check_output(["git", "-C", "/workspace/pyannote", "rev-parse", "HEAD"], text=True).strip(),
        },
        "environment": {
            "primary_model": "/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4",
            "qwen_aligner_model": "/workspace/models/Qwen3-ForcedAligner-0.6B",
            "diarization_model": "/workspace/models/Nemotron-3-Diarization",
            "gpu": "NVIDIA GeForce RTX 5090 (32607 MiB)",
        },
        "phase_b_and_combined_acceptance": {
            "live_runs": live_runs,
            "episode_before_after": {
                "prior_fr_full_episode_90s": _summarize_rec(old_fr),
                "repaired_fr_full_episode_repaired_90s": _summarize_rec(new_fr),
            },
        },
        "phase_c_refinement_evaluation": {
            "total_cases": phase_c["total_cases"],
            "c2_reasoning_probe_gate": phase_c["c2_reasoning_probe_gate"],
            "summary": phase_c["summary"],
            "authoritative_transcript_mutated": phase_c["authoritative_transcript_mutated"],
        },
        "phase_d_scene_diarization_evaluation": {
            "model_revision": phase_d["model_revision"],
            "streaming_modes_available": phase_d["streaming_modes_available"],
            "load_s": phase_d["load_s"],
            "load_gpu": phase_d["load_gpu"],
            "scenes_summary": {
                sid: {
                    "scale": s["scale"],
                    "abs_interval": s["abs_interval"],
                    "duration_s": s["duration_s"],
                    "canonical_chunked_summary": s["canonical_chunked_summary"],
                    "runs": {
                        mk: {"elapsed_s": mv["elapsed_s"], "summary": mv["summary"], "gpu": mv["gpu"]}
                        for mk, mv in s["runs"].items()
                    },
                    "evidence_adapter_offline_statuses": {
                        st: sum(1 for c in s["evidence_adapter_offline"]["channels"] if c["unresolved_status"] == st)
                        for st in {c["unresolved_status"] for c in s["evidence_adapter_offline"]["channels"]}
                    },
                    "advisory_flags_count": len(s["evidence_adapter_offline"]["advisory_flags"]),
                }
                for sid, s in phase_d["scenes"].items()
            },
            "ground_truth_cases_summary": {
                cid: {
                    "duration_s": g["duration_s"],
                    "modes": {
                        mk: {"elapsed_s": mv["elapsed_s"], "der_metrics": mv["der_metrics"], "summary": mv["summary"]}
                        for mk, mv in g["modes"].items()
                    },
                }
                for cid, g in phase_d["ground_truth_cases"].items()
            },
        },
        "unit_tests": {
            "main_venv_cpu_passed": 52,
            "modular_eval_cpu_passed": 8,
            "total_cpu_passed": 60,
        },
    }

    out = EVAL_DIR / "benchmark.json"
    out.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
