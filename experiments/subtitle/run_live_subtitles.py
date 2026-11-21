#!/usr/bin/env python3
"""Work Package B/C/D/E live GPU runs, profile comparisons, and FFmpeg visual verification."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

WORKSPACE = Path("/workspace/nvidia-transcribe")
if str(WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "src"))

from nvidia_transcribe.refinement import RefinementConfig, load_glossary, refine_utterances_with_swiftu
from nvidia_transcribe.subtitle_pipeline import SubtitlePipelineConfig, generate_subtitles
from nvidia_transcribe.subtitles import SubtitleConfig
from nvidia_transcribe.translation import TranslationConfig

OUT_ROOT = WORKSPACE / "outputs/subtitle_runs"
EPISODE_VIDEO = Path("/opt/mcog/Les mystérieuses cités d'or Saison 4 Episode 1 [YlVGUDNbyXk].webm")
EPISODE_CONTRACT = WORKSPACE / "outputs/reliability_runs/fr_full_episode_repaired_90s/recording.contract.json"
EPISODE_GLOSSARY = WORKSPACE / "configs/glossaries/mcog_fr.yaml"


def run_profile_comparison() -> dict[str, Any]:
    """Compare off / conservative / balanced / expressive refinement on 00:00-06:15 French excerpts."""
    c = json.loads(EPISODE_CONTRACT.read_text(encoding="utf-8"))
    subset = [u for u in (c.get("utterances") or []) if (u.get("source_end") or 0.0) <= 375.0]
    gl = load_glossary(EPISODE_GLOSSARY)

    out_dir = OUT_ROOT / "profile_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"utterance_count": len(subset), "profiles": {}}

    by_uid: dict[str, dict[str, Any]] = {}
    for u in subset:
        uid = str(u.get("id") or u.get("utterance_id"))
        by_uid[uid] = {
            "id": uid,
            "speaker": u.get("speaker_id"),
            "start": u.get("source_start"),
            "end": u.get("source_end"),
            "raw": u.get("source_text"),
        }

    for prof in ("off", "conservative", "balanced", "expressive"):
        ref_utts, meta = refine_utterances_with_swiftu(
            subset,
            source_language="fr",
            config=RefinementConfig(backend="swiftu", profile=prof, batch_size=6),
            glossary=gl,
            checkpoint_path=out_dir / f"{prof}.ckpt.json",
            resume=False,
        )
        report["profiles"][prof] = meta
        for u in ref_utts:
            uid = str(u.get("id") or u.get("utterance_id"))
            by_uid[uid][prof] = u.get("refined_text")
            by_uid[uid][f"{prof}_changed"] = u.get("refinement_applied")
            by_uid[uid][f"{prof}_type"] = u.get("refinement_edit_type")

    report["utterances"] = list(by_uid.values())
    (out_dir / "profile_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Profile comparison summary:", {p: (m["changed_count"], m.get("elapsed_s", 0.0)) for p, m in report["profiles"].items()})
    return report


def run_full_french_episode() -> dict[str, Any]:
    """Generate complete English SRT & ASS for French episode using balanced + swiftu."""
    out_dir = OUT_ROOT / "fr_full_episode"
    cfg = SubtitlePipelineConfig(
        source_language="fr",
        target_language="en",
        refinement=RefinementConfig(backend="swiftu", profile="balanced", batch_size=8),
        translation=TranslationConfig(backend="swiftu", style="natural", batch_size=8),
        subtitles=SubtitleConfig(max_chars=42, max_lines=2, min_display_s=1.2, max_display_s=6.5, overlap_threshold_s=0.4),
        formats=("srt", "ass"),
        episode_stem="Les mystérieuses cités d'or Saison 4 Episode 1 [YlVGUDNbyXk]",
    )
    meta = generate_subtitles(
        recording_contract_path=EPISODE_CONTRACT,
        output_dir=out_dir,
        config=cfg,
        glossary=EPISODE_GLOSSARY,
        progress_callback=lambda m: print(f"[fr-episode] {m}", flush=True),
    )
    print("Full FR episode subtitles generated:", meta["subtitles"])
    return meta


def run_mandarin_samples() -> dict[str, Any]:
    """Generate English SRT & ASS for verified repaired Mandarin contracts."""
    base_contracts = OUT_ROOT / "zh_qwen_fix_verify"
    gl_zh = load_glossary(["Scotturb", "Sintra", "Cape Roca", "calcium", "potassium"])
    samples = ("zh_fleurs_01_2006", "zh_fleurs_05_1805", "zh_2spk_overlap", "zh_multichunk_synthetic_qwen")
    results: dict[str, Any] = {}
    for s in samples:
        rc = base_contracts / s / "recording.contract.json"
        if not rc.is_file():
            raise FileNotFoundError(f"Run verify_qwen_fix_gpu.py first; missing {rc}")
        out_dir = OUT_ROOT / "zh_samples" / s
        cfg = SubtitlePipelineConfig(
            source_language="zh",
            target_language="en",
            refinement=RefinementConfig(backend="swiftu", profile="balanced", batch_size=6),
            translation=TranslationConfig(backend="swiftu", style="natural", batch_size=6),
            subtitles=SubtitleConfig(max_chars=42, max_lines=2, min_display_s=1.2, max_display_s=6.5, overlap_threshold_s=0.35),
            formats=("srt", "ass"),
            episode_stem=s,
        )
        meta = generate_subtitles(
            recording_contract_path=rc,
            output_dir=out_dir,
            config=cfg,
            glossary=gl_zh,
            progress_callback=lambda m, _s=s: print(f"[{_s}] {m}", flush=True),
        )
        results[s] = meta
    (OUT_ROOT / "zh_samples" / "mandarin_subtitles_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results


def render_visual_verification() -> dict[str, Any]:
    """Burn full-episode English ASS onto representative video timestamps via ffmpeg & inspect collisions."""
    ass_path = OUT_ROOT / "fr_full_episode" / "Les mystérieuses cités d'or Saison 4 Episode 1 [YlVGUDNbyXk].en.ass"
    translated = json.loads((OUT_ROOT / "fr_full_episode" / "translated.contract.json").read_text(encoding="utf-8"))
    shots_dir = OUT_ROOT / "visual_screenshots"
    shots_dir.mkdir(parents=True, exist_ok=True)

    # Pick timestamps dynamically: first active ASS lane-1 top+bottom overlap + opening/narrator + child voices F1 + fast F2 + later dialogue
    import re

    events = []
    for line in ass_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Dialogue:"):
            parts = line.split(",", 9)
            layer = int(parts[0].split(":")[1].strip())
            def _to_s(tstr: str) -> float:
                h, m, s = tstr.split(":")
                return int(h) * 3600 + int(m) * 60 + float(s)
            events.append({"start": _to_s(parts[1]), "end": _to_s(parts[2]), "style": parts[3], "layer": layer, "text": parts[9]})

    top_ev = [e for e in events if e["style"] == "Top"]
    timestamps: list[tuple[str, float, str]] = [
        ("shot_01_intro_theme", 10.0, "Opening scene / wide two-line Main caption"),
        ("shot_02_narrator", 47.5, "Narrator with proper nouns (Esteban, Zia, Tao)"),
        ("shot_03_child_voices_f1", 210.0, "Child voices (F1 region)"),
        ("shot_04_rapid_turns_f2", 268.0, "Fast dialogue (F2 region)"),
    ]
    if top_ev:
        # Take midpoint of first and a rich Top-lane simultaneous event
        e0 = top_ev[0]
        timestamps.append(("shot_05_simultaneous_overlap_1", round(0.5 * (e0["start"] + e0["end"]), 2), f"Simultaneous Top+Main lanes ({e0['text'][:50]})"))
        rich_top = next((e for e in top_ev[1:] if len(e["text"]) > 20), top_ev[min(len(top_ev) // 2, len(top_ev) - 1)])
        timestamps.append(("shot_06_simultaneous_overlap_2", round(0.5 * (rich_top["start"] + rich_top["end"]), 2), f"Simultaneous Top+Main lanes ({rich_top['text'][:50]})"))
    else:
        timestamps.append(("shot_05_overlap_o2", 351.0, "Dialogue O2 region"))
        timestamps.append(("shot_06_late_scene", 950.0, "Late episode scene"))

    rendered = []
    # Symlink with a clean ASCII filename inside shots_dir so libass/ffmpeg never trips on apostrophes/brackets
    import shutil
    safe_ass = shots_dir / "render_input.ass"
    shutil.copy2(ass_path, safe_ass)
    try:
        for name, ts_s, desc in timestamps:
            png_path = shots_dir / f"{name}_{int(round(ts_s))}s.png"
            pre_s = max(0.0, ts_s - 2.0)
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{pre_s:.3f}",
                "-copyts",
                "-i",
                str(EPISODE_VIDEO),
                "-vf",
                "ass=render_input.ass",
                "-ss",
                f"{ts_s:.3f}",
                "-frames:v",
                "1",
                str(png_path),
            ]
            subprocess.run(cmd, cwd=str(shots_dir), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            active = [e for e in events if e["start"] <= ts_s <= e["end"]]
            rendered.append({"name": name, "timestamp_s": ts_s, "description": desc, "png_path": str(png_path), "active_cues": active})
            print(f"Rendered {png_path.name} ({len(active)} active ASS cues: {[a['style'] for a in active]})")
    finally:
        safe_ass.unlink(missing_ok=True)

    (shots_dir / "visual_manifest.json").write_text(json.dumps(rendered, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"shots": rendered}


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else "--all-swiftu"
    if arg == "--profile-comparison":
        run_profile_comparison()
    elif arg == "--full-fr-episode":
        run_full_french_episode()
    elif arg == "--mandarin-samples":
        run_mandarin_samples()
    elif arg == "--render-screenshots":
        render_visual_verification()
    elif arg == "--all-swiftu":
        run_profile_comparison()
        run_full_french_episode()
        run_mandarin_samples()


if __name__ == "__main__":
    main()
