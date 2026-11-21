#!/usr/bin/env python3
"""Stage D comparative diagnostics and FFmpeg ASS screenshot generator for FR and ZH EP166 baselines."""

from __future__ import annotations

import json
import re
import shutil
import statistics
import subprocess
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

WORKSPACE = Path("/workspace/nvidia-transcribe")
FR_DIR = WORKSPACE / "outputs/subtitle_runs/fr_full_episode"
ZH_DIR = WORKSPACE / "outputs/subtitle_runs/zh_full_episode_ep166"
MEDIA = Path("/workspace/pyannote/outputs/sources/e4g3i65Kyf8/✨ A Will Eternal Final Season EP166 Full Version [MULTI SUB] [e4g3i65Kyf8].webm")
REF_EN_SRT = Path("/workspace/pyannote/outputs/sources/e4g3i65Kyf8/✨ A Will Eternal Final Season EP166 Full Version [MULTI SUB] [e4g3i65Kyf8].en.srt")
EVAL_DIR = WORKSPACE / "evaluations/mandarin-ep166"


def _parse_srt_ts(ts: str) -> float:
    h, m, rest = ts.strip().split(":")
    s, ms = rest.split(",")
    return int(h) * 3600.0 + int(m) * 60.0 + int(s) + int(ms) / 1000.0


def _parse_ass_ts(ts: str) -> float:
    h, m, s = ts.strip().split(":")
    return int(h) * 3600.0 + int(m) * 60.0 + float(s)


def parse_srt(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in re.split(r"\r?\n\s*\r?\n", raw.strip()) if b.strip()]
    cues: list[dict[str, Any]] = []
    for idx, blk in enumerate(blocks):
        lines = blk.splitlines()
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        s_str, e_str = [p.strip() for p in lines[1].split("-->", 1)]
        s_s = _parse_srt_ts(s_str)
        e_s = _parse_srt_ts(e_str)
        text_lines = [l.strip() for l in lines[2:] if l.strip()]
        cues.append({
            "index": idx + 1,
            "start": round(s_s, 3),
            "end": round(e_s, 3),
            "duration": round(e_s - s_s, 3),
            "lines": text_lines,
            "text": " ".join(text_lines),
        })
    return cues


def parse_ass(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        layer = int(parts[0].split(":", 1)[1].strip())
        s_s = _parse_ass_ts(parts[1])
        e_s = _parse_ass_ts(parts[2])
        style = parts[3].strip()
        speaker = parts[4].strip()
        raw_text = parts[9].strip()
        lines = [l.strip() for l in raw_text.split("\\N") if l.strip()]
        events.append({
            "layer": layer,
            "start": round(s_s, 3),
            "end": round(e_s, 3),
            "duration": round(e_s - s_s, 3),
            "style": style,
            "speaker": speaker,
            "lines": lines,
            "text": " ".join(lines),
        })
    return events


def _norm_txt(t: str) -> str:
    return re.sub(r"[^\w\s]", "", t.lower()).strip()


def analyze_episode(run_dir: Path, *, srt_name: str, ass_name: str, duration_s: float, lang: str) -> dict[str, Any]:
    tr = json.loads((run_dir / "translated.contract.json").read_text(encoding="utf-8"))
    sub_meta = json.loads((run_dir / "subtitles.metadata.json").read_text(encoding="utf-8"))
    srt_cues = parse_srt(run_dir / srt_name)
    ass_events = parse_ass(run_dir / ass_name)
    utts = tr.get("utterances") or []

    # 3-5 Timing checks on SRT (complete renderable track)
    invalid_srt = [c for c in srt_cues if c["start"] < 0 or c["end"] <= c["start"]]
    invalid_ass = [e for e in ass_events if e["start"] < 0 or e["end"] <= e["start"]]
    near_zero_srt = [c for c in srt_cues if c["duration"] < 0.2]
    under_1s_srt = [c for c in srt_cues if c["duration"] < 1.0]

    # 6-7 Reading speed distribution (only over positive-duration cues; invalid timing reported separately)
    cps_list: list[float] = []
    high_cps_examples: list[dict[str, Any]] = []
    for c in srt_cues:
        if c["duration"] <= 0.0:
            continue
        chars = len(c["text"])
        dur = c["duration"]
        cps = round(chars / dur, 2)
        cps_list.append(cps)
        if cps > 25.0:
            high_cps_examples.append({
                "index": c["index"],
                "start": c["start"],
                "end": c["end"],
                "duration": c["duration"],
                "cps": cps,
                "text": c["text"],
            })
    cps_sorted = sorted(cps_list)
    def _pct(p: float) -> float:
        if not cps_sorted:
            return 0.0
        k = min(len(cps_sorted) - 1, max(0, int(round(p * (len(cps_sorted) - 1)))))
        return round(cps_sorted[k], 2)

    # 8 Multi-line & excessive line length
    multiline = [c for c in srt_cues if len(c["lines"]) > 1]
    over_2_lines = [c for c in srt_cues if len(c["lines"]) > 2]
    over_42_chars = [c for c in srt_cues if any(len(l) > 42 for l in c["lines"])]

    # 9 Identical or highly similar concurrent cues
    concurrent_dups: list[dict[str, Any]] = []
    for i in range(len(srt_cues)):
        a = srt_cues[i]
        na = _norm_txt(a["text"])
        if len(na) < 4:
            continue
        for j in range(i + 1, len(srt_cues)):
            b = srt_cues[j]
            if b["start"] >= a["end"] - 0.05:
                break
            nb = _norm_txt(b["text"])
            if len(nb) < 4:
                continue
            sim = 1.0 if na == nb else SequenceMatcher(None, na, nb).ratio()
            if sim >= 0.80:
                concurrent_dups.append({
                    "cue_a": a["index"],
                    "cue_b": b["index"],
                    "start": max(a["start"], b["start"]),
                    "end": min(a["end"], b["end"]),
                    "similarity": round(sim, 3),
                    "text_a": a["text"],
                    "text_b": b["text"],
                })

    # 10-12 Overlaps & ASS omissions
    lane0 = [e for e in ass_events if e["style"] == "Main"]
    lane1 = [e for e in ass_events if e["style"] == "Top"]
    two_spk_overlaps = []
    for top in lane1:
        overlapping_main = [m for m in lane0 if m["start"] < top["end"] - 0.05 and top["start"] < m["end"] - 0.05]
        if overlapping_main:
            two_spk_overlaps.append({
                "top": top,
                "main": overlapping_main[0],
            })

    # Check ASS omissions by exact cue count & fuzzy 0.02s matching (ASS uses centisecond precision vs SRT ms)
    omitted_from_ass = []
    used_ass_indices: set[int] = set()
    for c in srt_cues:
        matched = False
        for idx_a, e in enumerate(ass_events):
            if idx_a in used_ass_indices:
                continue
            if e["text"] == c["text"] and abs(e["start"] - c["start"]) <= 0.025 and abs(e["end"] - c["end"]) <= 0.025:
                used_ass_indices.add(idx_a)
                matched = True
                break
        if not matched:
            omitted_from_ass.append(c)

    # 13 Translation omissions & source leakage + untimed/unrenderable utterances
    cjk_re = re.compile(r"[\u4e00-\u9fff]")
    fr_markers = re.compile(r"\b(est|nous|vous|c'est|les|des|dans|pour|avec|être|cité|d'or)\b", re.I)
    source_leaks: list[dict[str, Any]] = []
    for u in utts:
        if u.get("merged_into") or u.get("fragment_suppressed") or u.get("overlay_suppressed") or u.get("suppressed_duplicate_of"):
            continue
        t = str(u.get("translated_text") or "").strip()
        src = str(u.get("refined_text") or u.get("source_text") or "").strip()
        if not t:
            source_leaks.append({"id": u["id"], "reason": "empty_translation", "source": src})
        elif lang == "zh" and cjk_re.search(t):
            source_leaks.append({"id": u["id"], "reason": "cjk_in_english", "source": src, "translated": t})
        elif lang == "fr" and len(t.split()) >= 4 and t == src and fr_markers.search(t):
            source_leaks.append({"id": u["id"], "reason": "identical_french", "source": src, "translated": t})
        elif u.get("translation_status") not in ("translated", "translated_cached"):
            source_leaks.append({"id": u["id"], "reason": str(u.get("translation_status")), "source": src, "translated": t})

    untimed_utts = [u for u in utts if u.get("source_start") is None or u.get("source_end") is None]
    partial_utts = [u for u in utts if ((u.get("timestamp_provenance") or {}).get("timing_status") or u.get("timing_status")) == "partial"]
    # 14 Suspected timing drift: overly long raw utterance spans (> 12s) or split-mismatch > 10s
    long_raw_utts = [
        {
            "id": u["id"],
            "start": u.get("source_start"),
            "end": u.get("source_end"),
            "duration": round(float(u["source_end"]) - float(u["source_start"]), 3),
            "timing_status": (u.get("timestamp_provenance") or {}).get("timing_status") or u.get("timing_status"),
            "source_chars": len(str(u.get("source_text") or "")),
            "text": u.get("translated_text"),
        }
        for u in utts
        if u.get("source_start") is not None and u.get("source_end") is not None and (float(u["source_end"]) - float(u["source_start"])) > 12.0
    ]

    # 15 Rapid sequential cuts (<0.05s gap across different speakers in lane 0) & dense bursts
    rapid_swaps = []
    for i in range(1, len(srt_cues)):
        gap = srt_cues[i]["start"] - srt_cues[i - 1]["end"]
        if 0.0 <= gap <= 0.05 and srt_cues[i - 1]["duration"] < 1.5 and srt_cues[i]["duration"] < 1.5:
            rapid_swaps.append({"prev": srt_cues[i - 1]["index"], "next": srt_cues[i]["index"], "gap_s": round(gap, 3)})

    sub_summary = sub_meta.get("subtitles") or {}
    return {
        "language": lang,
        "duration_s": duration_s,
        "utterance_counts": {
            "total": len(utts),
            "merged_into_prev": sum(1 for u in utts if u.get("merged_into")),
            "suppressed_fragments_count": sum(1 for u in utts if u.get("fragment_suppressed")),
            "untimed_unrenderable": len(untimed_utts),
            "partial_timing": len(partial_utts),
        },
        "cue_counts": {
            "srt_cues": len(srt_cues),
            "ass_events": len(ass_events),
            "ass_main_cues": len(lane0),
            "ass_top_cues": len(lane1),
            "ass_overflow_cues": int(sub_summary.get("ass_overflow_cues") or len(omitted_from_ass)),
            "omitted_from_ass_count": len(omitted_from_ass),
            "omitted_from_ass": omitted_from_ass,
        },
        "timing_anomalies": {
            "invalid_ordering_srt": len(invalid_srt),
            "invalid_ordering_srt_examples": invalid_srt,
            "invalid_ordering_ass": len(invalid_ass),
            "zero_or_near_zero_lt_0_2s": len(near_zero_srt),
            "under_1_0s": len(under_1s_srt),
            "under_1_0s_examples": under_1s_srt[:10],
            "long_raw_utterances_gt_12s": long_raw_utts,
            "rapid_short_swaps_le_50ms": len(rapid_swaps),
        },
        "reading_speed_cps": {
            "min": round(min(cps_sorted), 2) if cps_sorted else 0.0,
            "p50": _pct(0.50),
            "p90": _pct(0.90),
            "p95": _pct(0.95),
            "max": round(max(cps_sorted), 2) if cps_sorted else 0.0,
            "mean": round(statistics.mean(cps_sorted), 2) if cps_sorted else 0.0,
            "gt_20_cps": sum(1 for x in cps_sorted if x > 20.0),
            "gt_25_cps": sum(1 for x in cps_sorted if x > 25.0),
            "gt_30_cps": sum(1 for x in cps_sorted if x > 30.0),
            "top_high_cps_examples": sorted(high_cps_examples, key=lambda x: -x["cps"])[:10],
        },
        "line_wrapping": {
            "multiline_cues": len(multiline),
            "over_2_lines": len(over_2_lines),
            "line_gt_42_chars": len(over_42_chars),
            "line_gt_42_examples": over_42_chars[:5],
        },
        "overlaps_and_duplicates": {
            "genuine_two_speaker_overlap_pairs": len(two_spk_overlaps),
            "concurrent_similar_or_duplicate_cues": len(concurrent_dups),
            "concurrent_duplicate_examples": concurrent_dups[:10],
        },
        "translation_integrity": {
            "source_leaks_or_omissions_count": len(source_leaks),
            "source_leaks_or_omissions": source_leaks,
        },
        "raw_cues_sample": {"srt_first_5": srt_cues[:5], "srt_last_5": srt_cues[-5:]},
    }


def render_ep166_screenshots(zh_diag: dict[str, Any]) -> list[dict[str, Any]]:
    ass_path = ZH_DIR / "A Will Eternal Final Season EP166 [e4g3i65Kyf8].en.ass"
    ass_events = parse_ass(ass_path)
    lane1 = [e for e in ass_events if e["style"] == "Top"]
    lane0 = [e for e in ass_events if e["style"] == "Main"]
    shots_dir = ZH_DIR / "review/screenshots"
    shots_dir.mkdir(parents=True, exist_ok=True)

    # Dynamically pick 10 representative + problematic frames from actual ASS cues
    picks: list[tuple[str, float, str]] = []

    def _add(name: str, ts: float, desc: str) -> None:
        picks.append((name, round(ts, 2), desc))

    # 1. Early opening cue (first rendered caption around ~101.8s)
    if lane0:
        c0 = lane0[0]
        _add("shot_01_opening_first_cue", 0.5 * (c0["start"] + c0["end"]), f"First timed English caption ({c0['text'][:40]})")
    # 2. Long unpunctuated recap narration / fast speech (~184s or ~250s where burned-in hardsubs are dense)
    recap_ev = next((e for e in lane0 if 180.0 <= e["start"] <= 190.0), None)
    if recap_ev:
        _add("shot_02_recap_hardsub_collision", 0.5 * (recap_ev["start"] + recap_ev["end"]), "Recap narration overlapping burned-in Chinese hardsubs")
    # 3. High-CPS cue (pick top high-CPS example that exists in ASS)
    top_cps = (zh_diag["reading_speed_cps"]["top_high_cps_examples"] or [None])[0]
    if top_cps:
        _add("shot_03_high_cps_cue", 0.5 * (top_cps["start"] + top_cps["end"]), f"High reading speed ({top_cps['cps']} cps)")
    # 4. Mid-episode cultivation lore dialogue (~380-455s, Underworld Emperor Palace / Tomb Keeper)
    lore_ev = next((e for e in lane0 if 405.0 <= e["start"] <= 460.0 and len(e["lines"]) == 2), None)
    if lore_ev:
        _add("shot_04_cultivation_lore_2line", 0.5 * (lore_ev["start"] + lore_ev["end"]), "Two-line cultivation exposition vs hardsubs")
    # 5. Concurrent duplicate / similar cue if present
    dups = zh_diag["overlaps_and_duplicates"]["concurrent_duplicate_examples"]
    if dups:
        d0 = dups[0]
        _add("shot_05_concurrent_similar_overlap", 0.5 * (d0["start"] + d0["end"]), f"Concurrent similar/echo overlap (sim={d0['similarity']})")
    # 6. Genuine two-speaker Top + Main overlap (#1)
    for t in lane1:
        m = next((x for x in lane0 if x["start"] < t["end"] - 0.1 and t["start"] < x["end"] - 0.1 and x["text"] != t["text"]), None)
        if m:
            mid = 0.5 * (max(t["start"], m["start"]) + min(t["end"], m["end"]))
            _add("shot_06_two_speaker_overlap_1", mid, "Genuine 2-speaker Top+Main separation")
            break
    # 7. Rapid conversation exchange / later action dialogue (~680-750s or 930-980s)
    rapid_ev = next((e for e in lane0 if 940.0 <= e["start"] <= 975.0), None)
    if rapid_ev:
        _add("shot_07_rapid_exchange", 0.5 * (rapid_ev["start"] + rapid_ev["end"]), "Rapid conversational turn")
    # 8. Genuine two-speaker overlap in 720-880s emotional / action climax
    for t in lane1:
        if t["start"] < 720.0:
            continue
        m = next((x for x in lane0 if x["start"] < t["end"] - 0.1 and t["start"] < x["end"] - 0.1 and x["text"] != t["text"]), None)
        if m:
            mid = 0.5 * (max(t["start"], m["start"]) + min(t["end"], m["end"]))
            _add("shot_08_emotional_2spk_overlap", mid, "Two-speaker overlap in emotional confrontation")
            break
    # 9. Emotional Bai Xiaochun confrontation (~1040-1100s)
    emo_ev = next((e for e in lane0 if 1040.0 <= e["start"] <= 1110.0 and len(e["text"]) > 28), None)
    if emo_ev:
        _add("shot_09_bai_xiaochun_confrontation", 0.5 * (emo_ev["start"] + emo_ev["end"]), "Late confrontation long two-line subtitle")
    # 10. Final timed subtitle near end of episode (~1195-1203s) + untimed Chunk 14 gap check frame at 1215s
    if lane0:
        c_last = lane0[-1]
        _add("shot_10_final_timed_cue", 0.5 * (c_last["start"] + c_last["end"]), f"Final timed subtitle ({c_last['text'][:40]})")
    _add("shot_11_untimed_chunk14_gap", 1212.0, "Active speech in Chunk 14 where English subtitle is missing (untimed alignment_failed)")

    safe_ass = shots_dir / "render_ep166.ass"
    shutil.copy2(ass_path, safe_ass)
    rendered: list[dict[str, Any]] = []
    try:
        for name, ts_s, desc in picks:
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
                    str(MEDIA),
                    "-vf",
                    "ass=render_ep166.ass",
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
            rendered.append({
                "name": name,
                "timestamp_s": ts_s,
                "description": desc,
                "png_path": str(png_path.relative_to(WORKSPACE)),
                "active_cues": [{"style": a["style"], "start": a["start"], "end": a["end"], "text": a["text"]} for a in active],
            })
            print(f"Rendered {png_path.name} ({len(active)} active cues: {[a['style'] for a in active]})")
    finally:
        safe_ass.unlink(missing_ok=True)
    return rendered


def main() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    fr_rc = json.loads((WORKSPACE / "outputs/reliability_runs/fr_full_episode_repaired_90s/recording.contract.json").read_text(encoding="utf-8"))
    fr_dur = float((fr_rc.get("source") or {}).get("duration_s") or 1379.88)
    zh_rc = json.loads((ZH_DIR / "asr/recording.contract.json").read_text(encoding="utf-8"))
    zh_dur = float((zh_rc.get("source") or {}).get("duration_s") or 1318.88)

    fr_diag = analyze_episode(
        FR_DIR,
        srt_name="Les mystérieuses cités d'or Saison 4 Episode 1 [YlVGUDNbyXk].en.srt",
        ass_name="Les mystérieuses cités d'or Saison 4 Episode 1 [YlVGUDNbyXk].en.ass",
        duration_s=fr_dur,
        lang="fr",
    )
    zh_diag = analyze_episode(
        ZH_DIR,
        srt_name="A Will Eternal Final Season EP166 [e4g3i65Kyf8].en.srt",
        ass_name="A Will Eternal Final Season EP166 [e4g3i65Kyf8].en.ass",
        duration_s=zh_dur,
        lang="zh",
    )

    shots = render_ep166_screenshots(zh_diag)

    # Optional read-only comparison summary against external reference .en.srt
    ref_cues = parse_srt(REF_EN_SRT) if REF_EN_SRT.is_file() else []
    ref_summary = {
        "present": bool(ref_cues),
        "path": str(REF_EN_SRT),
        "cue_count": len(ref_cues),
        "first_cue": ref_cues[0] if ref_cues else None,
        "last_cue": ref_cues[-1] if ref_cues else None,
        "cues_before_first_generated": sum(1 for c in ref_cues if c["end"] < (zh_diag["raw_cues_sample"]["srt_first_5"][0]["start"] if zh_diag["raw_cues_sample"]["srt_first_5"] else 0)),
        "cues_after_last_generated": sum(1 for c in ref_cues if c["start"] > (zh_diag["raw_cues_sample"]["srt_last_5"][-1]["end"] if zh_diag["raw_cues_sample"]["srt_last_5"] else 99999)),
    }

    combined = {
        "french_baseline": fr_diag,
        "mandarin_ep166": zh_diag,
        "screenshots": shots,
        "external_reference_srt_comparison": ref_summary,
    }

    (ZH_DIR / "review/timing-analysis.json").write_text(json.dumps(zh_diag, ensure_ascii=False, indent=2), encoding="utf-8")
    (ZH_DIR / "review/quality-summary.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    (EVAL_DIR / "timing-analysis.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== COMPARATIVE SUMMARY ===")
    for label, d in (("FR", fr_diag), ("ZH", zh_diag)):
        print(
            label,
            "utts=", d["utterance_counts"]["total"],
            "untimed=", d["utterance_counts"]["untimed_unrenderable"],
            "srt=", d["cue_counts"]["srt_cues"],
            "ass=", d["cue_counts"]["ass_events"],
            "ass_omit=", d["cue_counts"]["omitted_from_ass_count"],
            "lt1s=", d["timing_anomalies"]["under_1_0s"],
            "cps_p50/max=", d["reading_speed_cps"]["p50"], d["reading_speed_cps"]["max"],
            "cps>20/25/30=", (d["reading_speed_cps"]["gt_20_cps"], d["reading_speed_cps"]["gt_25_cps"], d["reading_speed_cps"]["gt_30_cps"]),
            "dups=", d["overlaps_and_duplicates"]["concurrent_similar_or_duplicate_cues"],
            "2spk=", d["overlaps_and_duplicates"]["genuine_two_speaker_overlap_pairs"],
            "overflow=", d["cue_counts"]["ass_overflow_cues"],
            "leaks=", d["translation_integrity"]["source_leaks_or_omissions_count"],
        )


if __name__ == "__main__":
    main()
