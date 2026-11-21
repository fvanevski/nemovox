#!/usr/bin/env python3
"""Run reproducible Nemotron 3.5 Transcribe NVFP4 evaluation suites against local vLLM server."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from nvidia_transcribe.client import InferenceRequestConfig, NemotronVLLMClient
from nvidia_transcribe.contract import to_pyannote_compatible_segments
from nvidia_transcribe.metrics import compute_cer, compute_wer

ROOT = Path("/workspace/nvidia-transcribe")
PYANNOTE_BASELINE_JSON = Path(
    "/workspace/pyannote/outputs/runs/20261120-215532-Les-myst-rieuses-cit-s-d-or-Saison-4-Episode-1-YlVGUDNbyXk/result.json"
)

FR_EXCERPTS = [
    {
        "clip_id": "fr_0316_0355_child_voices",
        "path": ROOT / "outputs/audio/fr/fr_0316_0355_child_voices.wav",
        "offset_s": 196.0,
        "duration_s": 39.0,
        "target": "Similar child voices and speaker identity consistency",
    },
    {
        "clip_id": "fr_0422_0439_rapid_turns",
        "path": ROOT / "outputs/audio/fr/fr_0422_0439_rapid_turns.wav",
        "offset_s": 262.0,
        "duration_s": 17.0,
        "target": "Rapid speaker changes and attribution confusion",
    },
    {
        "clip_id": "fr_0539_0542_overlap_nonlex",
        "path": ROOT / "outputs/audio/fr/fr_0539_0542_overlap_nonlex.wav",
        "offset_s": 339.0,
        "duration_s": 3.0,
        "target": "Speech overlapping a nonlexical vocalization",
    },
    {
        "clip_id": "fr_0550_0552_transition",
        "path": ROOT / "outputs/audio/fr/fr_0550_0552_transition.wav",
        "offset_s": 350.0,
        "duration_s": 2.0,
        "target": "Conversational transition",
    },
    {
        "clip_id": "fr_1031_1033_music_fp",
        "path": ROOT / "outputs/audio/fr/fr_1031_1033_music_fp.wav",
        "offset_s": 631.0,
        "duration_s": 2.0,
        "target": "Music-induced false-positive speaker activity",
    },
    {
        "clip_id": "fr_1731_1735_overlap_vox",
        "path": ROOT / "outputs/audio/fr/fr_1731_1735_overlap_vox.wav",
        "offset_s": 1051.0,
        "duration_s": 4.0,
        "target": "Overlapping vocalization",
    },
    {
        "clip_id": "fr_2208_2211_high_pitch",
        "path": ROOT / "outputs/audio/fr/fr_2208_2211_high_pitch.wav",
        "offset_s": 1328.0,
        "duration_s": 3.0,
        "target": "High-pitched speech and missed vocalization",
    },
    {
        "clip_id": "fr_0305_0445_cont_100s",
        "path": ROOT / "outputs/audio/fr/fr_0305_0445_cont_100s.wav",
        "offset_s": 185.0,
        "duration_s": 100.0,
        "target": "Continuous 100-second segment",
    },
]


def query_gpu_telemetry() -> dict[str, Any]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    parts = [p.strip() for p in out.split(",")]
    return {
        "gpu_name": parts[0],
        "memory_used_mib": int(parts[1]),
        "memory_free_mib": int(parts[2]),
        "memory_total_mib": int(parts[3]),
        "utilization_gpu_pct": int(float(parts[4])),
        "temperature_c": float(parts[5]),
        "power_draw_w": float(parts[6]),
    }


def extract_pyannote_slice(baseline: dict[str, Any], start_s: float, end_s: float) -> dict[str, Any]:
    utts = []
    speakers = set()
    for u in baseline.get("subtitle_utterances", []):
        us, ue = u.get("start"), u.get("end")
        if us is not None and ue is not None and ue >= start_s and us <= end_s:
            utts.append(
                {
                    "start": us,
                    "end": ue,
                    "speaker": u.get("speaker"),
                    "text": u.get("source_text"),
                    "word_count": u.get("word_count"),
                }
            )
            if u.get("speaker"):
                speakers.add(u.get("speaker"))
    return {
        "utterances": utts,
        "joined_text": " ".join(x["text"] for x in utts if x.get("text")),
        "speakers": sorted(speakers),
    }


def save_case(out_dir: Path, name: str, result: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.contract.json").write_text(
        json.dumps(result["contract"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / f"{name}.raw.json").write_text(
        json.dumps(result["raw_response"], ensure_ascii=False, indent=2), encoding="utf-8"
    )


def summarize_contract(res: dict[str, Any], duration_s: float) -> dict[str, Any]:
    c = res["contract"]
    prov = c.get("model_provenance") or {}
    tprov = c.get("timestamp_provenance") or {}
    utts = c.get("utterances") or []
    words = [w for u in utts for w in (u.get("source_words") or []) + (u.get("source_characters") or [])]
    valid_words = [w for w in words if w.get("valid_timing")]
    speakers = sorted({u.get("speaker_id") for u in utts if u.get("speaker_id")})
    lat = res["latency_s"]
    return {
        "latency_s": lat,
        "audio_duration_s": duration_s,
        "rtf": round(lat / duration_s, 4) if duration_s > 0 else None,
        "x_realtime": round(duration_s / lat, 2) if lat > 0 else None,
        "finish_reason": prov.get("finish_reason"),
        "usage": prov.get("usage"),
        "raw_text": c.get("raw_text"),
        "source_text": c.get("source_text"),
        "english_translation": c.get("english_translation"),
        "speaker_count_transcript": len(speakers),
        "speakers_transcript": speakers,
        "speaker_mapping": c.get("speaker_mapping"),
        "diarization_segments": len(c.get("diarization_intervals") or []),
        "total_units": len(words),
        "valid_timed_units": len(valid_words),
        "ctc_error": tprov.get("error"),
        "invalid_timing_count": tprov.get("invalid_timing_count"),
    }


def run_all(base_url: str, model: str, run_longform: bool = True) -> dict[str, Any]:
    client = NemotronVLLMClient(base_url=base_url, model=model, timeout_s=1800.0)
    resp_dir = ROOT / "outputs/responses"
    bench: dict[str, Any] = {
        "model": model,
        "base_url": base_url,
        "gpu_before": query_gpu_telemetry(),
        "warmup": {},
        "french": {},
        "mandarin": {},
        "overlap": {},
        "longform": {},
    }

    # 1. Warmup (excluded from steady-state measurements)
    warmup_clip = FR_EXCERPTS[3]  # 2s transition
    w_res = client.transcribe(
        warmup_clip["path"],
        InferenceRequestConfig(mode="speaker", capture_ctc_timestamps=True, source_offset_s=warmup_clip["offset_s"]),
    )
    bench["warmup"] = summarize_contract(w_res, warmup_clip["duration_s"])
    bench["gpu_after_warmup"] = query_gpu_telemetry()

    # 2. French corpus vs Pyannote baseline
    baseline = json.loads(PYANNOTE_BASELINE_JSON.read_text(encoding="utf-8"))
    for ex in FR_EXCERPTS:
        cid = ex["clip_id"]
        off = ex["offset_s"]
        dur = ex["duration_s"]
        py_slice = extract_pyannote_slice(baseline, off, off + dur)

        res_spk = client.transcribe(
            ex["path"],
            InferenceRequestConfig(mode="speaker", source_language=None, capture_ctc_timestamps=True, source_offset_s=off),
        )
        save_case(resp_dir / "fr", f"{cid}_speaker", res_spk)

        res_verb = client.transcribe(
            ex["path"],
            InferenceRequestConfig(mode="verbatim", source_language=None, capture_ctc_timestamps=False, source_offset_s=off),
        )
        save_case(resp_dir / "fr", f"{cid}_verbatim", res_verb)

        res_lid = client.transcribe(
            ex["path"],
            InferenceRequestConfig(mode="lid", capture_ctc_timestamps=False),
        )

        entry: dict[str, Any] = {
            "target": ex["target"],
            "offset_s": off,
            "duration_s": dur,
            "lid": (res_lid["contract"].get("raw_text") or "").strip(),
            "speaker_ctc": summarize_contract(res_spk, dur),
            "verbatim": summarize_contract(res_verb, dur),
            "pyannote_baseline": py_slice,
            "wer_vs_pyannote_verbatim": compute_wer(py_slice["joined_text"], res_verb["contract"].get("source_text") or ""),
        }

        if cid in ("fr_0539_0542_overlap_nonlex", "fr_1031_1033_music_fp", "fr_1731_1735_overlap_vox", "fr_2208_2211_high_pitch"):
            res_rich = client.transcribe(
                ex["path"],
                InferenceRequestConfig(mode="rich", capture_ctc_timestamps=True, source_offset_s=off),
            )
            save_case(resp_dir / "fr", f"{cid}_rich", res_rich)
            entry["rich"] = summarize_contract(res_rich, dur)

        if cid in ("fr_0316_0355_child_voices", "fr_0422_0439_rapid_turns", "fr_0305_0445_cont_100s"):
            res_tr = client.transcribe(
                ex["path"],
                InferenceRequestConfig(mode="translate", target_language="English", capture_ctc_timestamps=False),
            )
            save_case(resp_dir / "fr", f"{cid}_translate_en", res_tr)
            entry["translate_en"] = summarize_contract(res_tr, dur)

        bench["french"][cid] = entry

    # 3. Mandarin Chinese FLEURS corpus (auto vs explicit language + timing + English translation)
    zh_manifest = json.loads((ROOT / "outputs/audio/zh/zh_manifest.json").read_text(encoding="utf-8"))
    zh_auto_edits, zh_auto_chars = 0, 0
    zh_exp_edits, zh_exp_chars = 0, 0
    for item in zh_manifest:
        cid = item["clip_id"]
        dur = item["duration_s"]
        ref = item["raw_transcription"]

        res_auto = client.transcribe(
            item["wav_path"],
            InferenceRequestConfig(mode="fleurs", source_language=None, capture_ctc_timestamps=True),
        )
        save_case(resp_dir / "zh", f"{cid}_auto", res_auto)

        res_exp = client.transcribe(
            item["wav_path"],
            InferenceRequestConfig(mode="fleurs", source_language="Mandarin Chinese", capture_ctc_timestamps=True),
        )
        save_case(resp_dir / "zh", f"{cid}_explicit", res_exp)

        res_tr = client.transcribe(
            item["wav_path"],
            InferenceRequestConfig(mode="translate", target_language="English", capture_ctc_timestamps=False),
        )
        save_case(resp_dir / "zh", f"{cid}_translate_en", res_tr)

        res_lid = client.transcribe(
            item["wav_path"],
            InferenceRequestConfig(mode="lid", capture_ctc_timestamps=False),
        )

        cer_auto = compute_cer(ref, res_auto["contract"].get("source_text") or "")
        cer_exp = compute_cer(ref, res_exp["contract"].get("source_text") or "")
        zh_auto_edits += cer_auto["edits"]
        zh_auto_chars += cer_auto["ref_chars"]
        zh_exp_edits += cer_exp["edits"]
        zh_exp_chars += cer_exp["ref_chars"]

        bench["mandarin"][cid] = {
            "fleurs_id": item["fleurs_id"],
            "gender": item["gender"],
            "duration_s": dur,
            "reference": ref,
            "lid": (res_lid["contract"].get("raw_text") or "").strip(),
            "auto_ctc": summarize_contract(res_auto, dur),
            "explicit_ctc": summarize_contract(res_exp, dur),
            "translate_en": summarize_contract(res_tr, dur),
            "cer_auto": cer_auto,
            "cer_explicit": cer_exp,
        }

    bench["mandarin_summary"] = {
        "clips": len(zh_manifest),
        "total_ref_chars": zh_auto_chars,
        "corpus_cer_auto": round(zh_auto_edits / max(zh_auto_chars, 1), 4),
        "corpus_cer_explicit": round(zh_exp_edits / max(zh_exp_chars, 1), 4),
    }

    # 4. French FLEURS clean ASR + controlled overlap fixtures
    fr_fleurs = json.loads((ROOT / "outputs/audio/overlap/fr_fleurs_manifest.json").read_text(encoding="utf-8"))
    fr_edits, fr_words = 0, 0
    fr_fleurs_res = {}
    for item in fr_fleurs:
        cid = item["clip_id"]
        dur = item["duration_s"]
        ref = item["raw_transcription"]
        res = client.transcribe(
            item["wav_path"],
            InferenceRequestConfig(mode="fleurs", source_language=None, capture_ctc_timestamps=True),
        )
        save_case(resp_dir / "fr_fleurs", cid, res)
        wer = compute_wer(ref, res["contract"].get("source_text") or "")
        fr_edits += wer["edits"]
        fr_words += wer["ref_words"]
        fr_fleurs_res[cid] = {
            "duration_s": dur,
            "reference": ref,
            "result": summarize_contract(res, dur),
            "wer": wer,
        }
    bench["french_fleurs"] = {
        "clips": fr_fleurs_res,
        "corpus_wer": round(fr_edits / max(fr_words, 1), 4),
        "total_ref_words": fr_words,
    }

    ov_manifest = json.loads((ROOT / "outputs/audio/overlap/overlap_manifest.json").read_text(encoding="utf-8"))
    for name, ov in ov_manifest.items():
        dur = ov["total_duration_s"]
        res_spk = client.transcribe(
            ov["wav_path"],
            InferenceRequestConfig(mode="speaker", capture_ctc_timestamps=True),
        )
        res_sot = client.transcribe(
            ov["wav_path"],
            InferenceRequestConfig(mode="multispeaker_sot", capture_ctc_timestamps=True),
        )
        save_case(resp_dir / "overlap", f"{name}_speaker", res_spk)
        save_case(resp_dir / "overlap", f"{name}_sot", res_sot)
        bench["overlap"][name] = {
            "fixture": ov,
            "speaker_mode": summarize_contract(res_spk, dur),
            "multispeaker_sot_mode": summarize_contract(res_sot, dur),
        }

    # 5. Longer-form progressive evaluation (5 min excerpt; full episode if feasible)
    if run_longform:
        for lf_name, lf_path, lf_dur in [
            ("fr_0000_0500_300s", ROOT / "outputs/audio/fr/fr_0000_0500_300s.wav", 300.0),
        ]:
            if lf_path.is_file():
                res_lf = client.transcribe(
                    lf_path,
                    InferenceRequestConfig(mode="speaker", capture_ctc_timestamps=True, source_offset_s=0.0),
                )
                save_case(resp_dir / "longform", lf_name, res_lf)
                bench["longform"][lf_name] = {
                    "summary": summarize_contract(res_lf, lf_dur),
                    "gpu_after": query_gpu_telemetry(),
                }

    # 6. Export one pyannote-compatible sample and test read-only subtitle writer if available
    sample_contract = json.loads((resp_dir / "fr/fr_0316_0355_child_voices_speaker.contract.json").read_text(encoding="utf-8"))
    py_segs = to_pyannote_compatible_segments(sample_contract)
    (ROOT / "outputs/renders/fr_0316_0355_pyannote_segments.json").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs/renders/fr_0316_0355_pyannote_segments.json").write_text(
        json.dumps(py_segs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    bench["gpu_final"] = query_gpu_telemetry()
    out_bench = ROOT / "evaluations/nemotron35/benchmark.json"
    out_bench.parent.mkdir(parents=True, exist_ok=True)
    out_bench.write_text(json.dumps(bench, ensure_ascii=False, indent=2), encoding="utf-8")
    return bench


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8190/v1")
    p.add_argument("--model", default="/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4")
    p.add_argument("--no-longform", action="store_true")
    args = p.parse_args()
    bench = run_all(args.base_url, args.model, run_longform=not args.no_longform)
    print("Benchmark complete. Mandarin summary:", json.dumps(bench["mandarin_summary"], ensure_ascii=False))
    print("French FLEURS corpus WER:", bench["french_fleurs"]["corpus_wer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
