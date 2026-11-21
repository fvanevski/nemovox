#!/usr/bin/env python3
"""Run regenerated FR & Mandarin EP166 viewing-quality pipelines, comparison diagnostics, and screenshots."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

WORKSPACE = Path("/workspace/nvidia-transcribe")
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
if str(WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "src"))

from experiments.subtitle.analyze_ep166_baseline import analyze_episode, parse_ass
from nvidia_transcribe.refinement import RefinementConfig
from nvidia_transcribe.subtitle_pipeline import SubtitlePipelineConfig, generate_subtitles, _atomic_write, _format_utterance_transcript_lines
from nvidia_transcribe.subtitles import SubtitleConfig, render_subtitles
from nvidia_transcribe.translation import TranslationConfig, condense_high_cps_cues_with_swiftu

VQ_ROOT = WORKSPACE / "outputs/subtitle_runs/viewing_quality"
BASE_FR = WORKSPACE / "outputs/subtitle_runs/fr_full_episode"
BASE_ZH = WORKSPACE / "outputs/subtitle_runs/zh_full_episode_ep166"

FR_VIDEO = Path("/opt/mcog/Les mystérieuses cités d'or Saison 4 Episode 1 [YlVGUDNbyXk].webm")
ZH_VIDEO = Path("/workspace/pyannote/outputs/sources/e4g3i65Kyf8/✨ A Will Eternal Final Season EP166 Full Version [MULTI SUB] [e4g3i65Kyf8].webm")
EVAL_DIR = WORKSPACE / "evaluations/viewing-quality"

FR_STEM = "Les mystérieuses cités d'or Saison 4 Episode 1 [YlVGUDNbyXk]"
ZH_STEM = "A Will Eternal Final Season EP166 [e4g3i65Kyf8]"


def regenerate_french() -> dict[str, Any]:
    """Regenerate FR subtitles from frozen translated baseline + selective condensation + C2/C4 fixes."""
    out_dir = VQ_ROOT / "fr_full_episode"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_tr = json.loads((BASE_FR / "translated.contract.json").read_text(encoding="utf-8"))

    condensed_utts, cond_summary = condense_high_cps_cues_with_swiftu(
        base_tr["utterances"],
        source_language="fr",
        config=TranslationConfig(backend="swiftu", style="natural"),
        cps_threshold=25.0,
        target_cps=18.0,
        checkpoint_path=out_dir / "condensation.checkpoint.json",
        resume=True,
    )
    translated_contract = {
        **base_tr,
        "utterances": condensed_utts,
        "condensation": cond_summary,
    }
    _atomic_write(out_dir / "translated.contract.json", json.dumps(translated_contract, ensure_ascii=False, indent=2))
    _atomic_write(out_dir / "raw_source_transcript.txt", _format_utterance_transcript_lines(condensed_utts, "raw_source_text", "source_text"))
    _atomic_write(out_dir / "refined_source_transcript.txt", _format_utterance_transcript_lines(condensed_utts, "refined_text", "source_text"))
    _atomic_write(out_dir / "final_english_subtitles.txt", _format_utterance_transcript_lines(condensed_utts, "translated_text"))

    cfg = SubtitleConfig(
        max_chars=42,
        max_lines=2,
        min_display_s=1.2,
        max_display_s=6.5,
        target_cps=18.0,
        overlap_threshold_s=0.4,
        ass_layout="standard",
        ass_max_lanes=3,
    )
    render_res = render_subtitles(
        translated_contract,
        output_dir=out_dir,
        episode_stem=FR_STEM,
        target_language="en",
        formats=("srt", "ass"),
        config=cfg,
    )
    meta = {
        "episode_stem": FR_STEM,
        "source_language": "fr",
        "target_language": "en",
        "condensation": cond_summary,
        "subtitles": render_res["stats"],
        "subtitle_conflicts": render_res["conflicts"][:200],
    }
    _atomic_write(out_dir / "subtitles.metadata.json", json.dumps(meta, ensure_ascii=False, indent=2))
    print("[VQ-FR] Done:", meta["subtitles"], "condensed:", cond_summary["condensed_count"], flush=True)
    return meta


def regenerate_mandarin() -> dict[str, Any]:
    """Regenerate Mandarin EP166 from frozen ASR contract with A1-A3, B, C1-C4, and D raised_main."""
    out_dir = VQ_ROOT / "zh_full_episode_ep166"
    cfg = SubtitlePipelineConfig(
        source_language="zh",
        target_language="en",
        refinement=RefinementConfig(backend="swiftu", profile="balanced", batch_size=6),
        translation=TranslationConfig(backend="swiftu", style="natural", batch_size=6),
        subtitles=SubtitleConfig(
            max_chars=42,
            max_lines=2,
            min_display_s=1.2,
            max_display_s=6.5,
            target_cps=18.0,
            overlap_threshold_s=0.4,
            ass_layout="raised_main",
            raised_main_margin_v=150,
            ass_max_lanes=3,
        ),
        formats=("srt", "ass"),
        episode_stem=ZH_STEM,
        condense_high_cps=True,
        condense_cps_threshold=25.0,
        reconstruct_untimed_order=True,
    )
    meta = generate_subtitles(
        recording_contract_path=BASE_ZH / "asr/recording.contract.json",
        output_dir=out_dir,
        config=cfg,
        glossary=WORKSPACE / "configs/glossaries/a_will_eternal_zh.yaml",
        progress_callback=lambda m: print(f"[VQ-ZH] {m}", flush=True),
    )
    print("[VQ-ZH] Done:", meta["subtitles"], "condensation:", meta.get("condensation"), flush=True)
    return meta


def render_ffmpeg_screenshots(
    *,
    video_path: Path,
    ass_path: Path,
    shots_dir: Path,
    timestamps: list[tuple[str, float, str]],
) -> list[dict[str, Any]]:
    shots_dir.mkdir(parents=True, exist_ok=True)
    ass_events = parse_ass(ass_path)
    safe_ass = shots_dir / "render_input.ass"
    shutil.copy2(ass_path, safe_ass)
    out: list[dict[str, Any]] = []
    try:
        for name, ts_s, desc in timestamps:
            png_path = shots_dir / f"{name}_{int(round(ts_s))}s.png"
            pre_s = max(0.0, ts_s - 2.0)
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{pre_s:.3f}",
                    "-copyts",
                    "-i",
                    str(video_path),
                    "-vf",
                    "ass=render_input.ass",
                    "-ss",
                    f"{ts_s:.3f}",
                    "-frames:v",
                    "1",
                    str(png_path),
                ],
                cwd=str(shots_dir),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            active = [e for e in ass_events if e["start"] <= ts_s <= e["end"]]
            out.append({
                "name": name,
                "timestamp_s": ts_s,
                "description": desc,
                "png_path": str(png_path.relative_to(WORKSPACE)),
                "active_cues": [{"style": a["style"], "start": a["start"], "end": a["end"], "text": a["text"]} for a in active],
            })
            print(f"Screenshot {png_path.name}: {[a['style'] for a in active]}")
    finally:
        safe_ass.unlink(missing_ok=True)
    return out


def compute_and_save_all() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    fr_before = analyze_episode(BASE_FR, srt_name=f"{FR_STEM}.en.srt", ass_name=f"{FR_STEM}.en.ass", duration_s=1379.88, lang="fr")
    fr_after = analyze_episode(VQ_ROOT / "fr_full_episode", srt_name=f"{FR_STEM}.en.srt", ass_name=f"{FR_STEM}.en.ass", duration_s=1379.88, lang="fr")
    zh_before = analyze_episode(BASE_ZH, srt_name=f"{ZH_STEM}.en.srt", ass_name=f"{ZH_STEM}.en.ass", duration_s=1318.88, lang="zh")
    zh_after = analyze_episode(VQ_ROOT / "zh_full_episode_ep166", srt_name=f"{ZH_STEM}.en.srt", ass_name=f"{ZH_STEM}.en.ass", duration_s=1318.88, lang="zh")

    # Render required Mandarin EP166 timestamps (102, 181, 270, 368, 411, 729, 942, 1045, 1201 + 3-way overlap)
    zh_ass = VQ_ROOT / "zh_full_episode_ep166" / f"{ZH_STEM}.en.ass"
    zh_events = parse_ass(zh_ass)
    zh_picks: list[tuple[str, float, str]] = [
        ("zh_01_102s_opening_hardsub", 102.0, "Opening scene raised Main above Chinese hardsub"),
        ("zh_02_181s_recap_hardsub", 181.2, "Recap caption raised cleanly above Chinese hardsub"),
        ("zh_03_270s_echo_fix", 270.0, "Formerly 153-CPS echo cue utt-0016/0017 region"),
        ("zh_04_368s_two_spk", 368.0, "Two-speaker Main + Top overlap"),
        ("zh_05_411s_lore_two_spk", 411.0, "Cultivation lore 2-line + Top separation"),
        ("zh_06_729s_emotional", 729.0, "Emotional confrontation"),
        ("zh_07_942s_rapid", 942.0, "Rapid exchange"),
        ("zh_08_1045s_confrontation", 1045.0, "Bai Xiaochun confrontation raised above hardsub"),
        ("zh_09_1201s_final", 1201.0, "Final dialogue above hardsub"),
    ]
    top_alt = next((e for e in zh_events if e["style"] == "TopAlt"), None)
    if top_alt:
        zh_picks.append(("zh_10_topalt_3spk", round(0.5 * (top_alt["start"] + top_alt["end"]), 2), "3-speaker Main+Top+TopAlt check"))

    zh_shots = render_ffmpeg_screenshots(
        video_path=ZH_VIDEO,
        ass_path=zh_ass,
        shots_dir=VQ_ROOT / "zh_full_episode_ep166/screenshots",
        timestamps=zh_picks,
    )

    fr_ass = VQ_ROOT / "fr_full_episode" / f"{FR_STEM}.en.ass"
    fr_picks: list[tuple[str, float, str]] = [
        ("fr_01_48s_narrator", 47.5, "Standard layout narrator"),
        ("fr_02_175s_2spk", 175.2, "Standard Main+Top 2-speaker overlap"),
        ("fr_03_284s_dup_fix", 283.58, "Formerly duplicate overlap at 282-291s"),
        ("fr_04_306s_3spk_topalt", 306.0, "3-speaker scene preserved via TopAlt"),
    ]
    fr_shots = render_ffmpeg_screenshots(
        video_path=FR_VIDEO,
        ass_path=fr_ass,
        shots_dir=VQ_ROOT / "fr_full_episode/screenshots",
        timestamps=fr_picks,
    )

    fr_meta = json.loads((VQ_ROOT / "fr_full_episode/subtitles.metadata.json").read_text(encoding="utf-8"))
    zh_meta = json.loads((VQ_ROOT / "zh_full_episode_ep166/subtitles.metadata.json").read_text(encoding="utf-8"))
    summary = {
        "french": {"before": fr_before, "after": fr_after, "condensation": fr_meta.get("condensation"), "screenshots": fr_shots},
        "mandarin": {"before": zh_before, "after": zh_after, "refinement": zh_meta.get("refinement"), "translation": zh_meta.get("translation"), "condensation": zh_meta.get("condensation"), "screenshots": zh_shots},
    }
    (EVAL_DIR / "comparison_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== BEFORE vs AFTER METRICS ===")
    for lang, b, a in (("FR", fr_before, fr_after), ("ZH", zh_before, zh_after)):
        print(f"{lang} BEFORE: srt={b['cue_counts']['srt_cues']} ass={b['cue_counts']['ass_events']} inv={b['timing_anomalies']['invalid_ordering_srt']} <1s={b['timing_anomalies']['under_1_0s']} >20={b['reading_speed_cps']['gt_20_cps']} >30={b['reading_speed_cps']['gt_30_cps']} max={b['reading_speed_cps']['max']} dups={b['overlaps_and_duplicates']['concurrent_similar_or_duplicate_cues']} omit={b['cue_counts']['omitted_from_ass_count']} leaks={b['translation_integrity']['source_leaks_or_omissions_count']}")
        print(f"{lang} AFTER : srt={a['cue_counts']['srt_cues']} ass={a['cue_counts']['ass_events']} inv={a['timing_anomalies']['invalid_ordering_srt']} <1s={a['timing_anomalies']['under_1_0s']} >20={a['reading_speed_cps']['gt_20_cps']} >30={a['reading_speed_cps']['gt_30_cps']} max={a['reading_speed_cps']['max']} dups={a['overlaps_and_duplicates']['concurrent_similar_or_duplicate_cues']} omit={a['cue_counts']['omitted_from_ass_count']} leaks={a['translation_integrity']['source_leaks_or_omissions_count']}")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--wait-server", action="store_true")
    p.add_argument("--only-analyze", action="store_true")
    args = p.parse_args()
    if args.only_analyze:
        compute_and_save_all()
        return
    if args.wait_server:
        import httpx
        for _ in range(90):
            try:
                if httpx.get("http://127.0.0.1:8001/v1/models", timeout=2.0).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.5)
    regenerate_french()
    regenerate_mandarin()
    compute_and_save_all()


if __name__ == "__main__":
    main()
