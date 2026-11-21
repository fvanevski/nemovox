#!/usr/bin/env python3
"""Experiment D: Conditional modular & hybrid integration probe on 5-min French passage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.modular.common import MODULAR_OUT, WORKSPACE_ROOT


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def align_words_in_windows_subproc_hint() -> None:
    """Placeholder documentation: alignment items come precomputed per <=150s window in raw JSON."""


def attribute_words_with_diarization(
    words: list[dict[str, Any]],
    diar_segments: list[dict[str, Any]],
    *,
    edge_duration_s: float = 0.18,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign speaker IDs from continuous Nemotron-3-Diarization evidence without violating invariants.

    Invariants enforced:
    - Exact word order and text preserved verbatim.
    - Unattributed words remain 'unknown' (never forced to speaker_0).
    - Simultaneous active diarization speakers are preserved in overlap metadata, never collapsed.
    - Short boundary words (<0.18s) do not use naive midpoint proximity alone.
    """
    attributed: list[dict[str, Any]] = []
    stats = {
        "total_words": len(words),
        "attributed_words": 0,
        "unknown_words": 0,
        "zero_duration_words": 0,
        "overlap_flagged_words": 0,
        "boundary_tiebreak_words": 0,
    }

    # First pass: raw evidence per word
    raw_matches: list[dict[str, Any]] = []
    for w in words:
        st = w.get("start")
        en = w.get("end")
        if st is None or en is None:
            raw_matches.append({"overlap": {}, "zero_dur": False, "unaligned": True})
            continue
        st_f, en_f = float(st), float(en)
        is_zero = abs(en_f - st_f) <= 1e-6
        if is_zero:
            stats["zero_duration_words"] += 1
        q0, q1 = (st_f - 0.08, en_f + 0.08) if is_zero else (st_f, en_f)
        by_spk: dict[str, float] = {}
        for seg in diar_segments:
            ov = _overlap(q0, q1, float(seg["start"]), float(seg["end"]))
            if ov > 0.0:
                sp = f"SPK_{int(seg['speaker']):02d}"
                by_spk[sp] = by_spk.get(sp, 0.0) + ov
        raw_matches.append({"overlap": by_spk, "zero_dur": is_zero, "unaligned": False})

    # Second pass: resolve primary + simultaneous speakers with edge-word context
    for idx, (w, m) in enumerate(zip(words, raw_matches)):
        by_spk = m["overlap"]
        st = w.get("start")
        en = w.get("end")
        dur = 0.0 if (st is None or en is None) else max(0.0, float(en) - float(st))
        ranked = sorted(by_spk.items(), key=lambda kv: (-kv[1], kv[0]))

        if not ranked:
            chosen = "unknown"
            concurrent: list[str] = []
            conf = "no_diarization_overlap"
            stats["unknown_words"] += 1
        elif len(ranked) == 1:
            chosen = ranked[0][0]
            concurrent = [chosen]
            conf = "unique_overlap"
            stats["attributed_words"] += 1
        else:
            top_sp, top_ov = ranked[0]
            sec_sp, sec_ov = ranked[1]
            concurrent = [sp for sp, ov in ranked if ov >= 0.35 * top_ov and ov >= 0.04]
            if len(concurrent) >= 2:
                stats["overlap_flagged_words"] += 1
            if abs(top_ov - sec_ov) <= 0.03:
                # Near tie or transition boundary: check short-word neighborhood instead of midpoint
                prev_cands = sorted(raw_matches[idx - 1]["overlap"].items(), key=lambda kv: -kv[1]) if idx > 0 else []
                next_cands = sorted(raw_matches[idx + 1]["overlap"].items(), key=lambda kv: -kv[1]) if idx + 1 < len(words) else []
                prev_sp = prev_cands[0][0] if prev_cands else None
                next_sp = next_cands[0][0] if next_cands else None
                if dur <= edge_duration_s and prev_sp in (top_sp, sec_sp) and next_sp == prev_sp:
                    chosen = prev_sp
                    conf = "boundary_clause_continuity"
                    stats["boundary_tiebreak_words"] += 1
                    stats["attributed_words"] += 1
                else:
                    chosen = "unknown"
                    conf = "ambiguous_boundary_tie"
                    stats["unknown_words"] += 1
            else:
                chosen = top_sp
                conf = "dominant_overlap_with_simultaneous" if len(concurrent) >= 2 else "dominant_overlap"
                stats["attributed_words"] += 1

        attributed.append(
            {
                "text": w.get("text", ""),
                "start": st,
                "end": en,
                "speaker": chosen,
                "concurrent_speakers": concurrent,
                "zero_duration": m["zero_dur"],
                "unaligned": m["unaligned"],
                "attribution_reason": conf,
            }
        )

    return attributed, stats


def build_experimental_contract(
    *,
    variant: str,
    source_wav: str,
    duration_s: float,
    source_offset_s: float,
    source_text: str,
    attributed_words: list[dict[str, Any]],
    diar_segments: list[dict[str, Any]],
    stats: dict[str, Any],
) -> dict[str, Any]:
    """Assemble utterances without dropping unaligned words or merging concurrent speaker regions."""
    utterances: list[dict[str, Any]] = []
    cur_words: list[dict[str, Any]] = []
    cur_spk: str | None = None

    def flush() -> None:
        nonlocal cur_words, cur_spk
        if not cur_words:
            return
        timed = [w for w in cur_words if w["start"] is not None and w["end"] is not None]
        st = min(w["start"] for w in timed) if timed else None
        en = max(w["end"] for w in timed) if timed else None
        conc = sorted({s for w in cur_words for s in w["concurrent_speakers"] if s != "unknown"})
        utterances.append(
            {
                "id": f"utt-{len(utterances):04d}",
                "speaker": cur_spk or "unknown",
                "start": st,
                "end": en,
                "text": " ".join(w["text"] for w in cur_words if w["text"]),
                "word_count": len(cur_words),
                "concurrent_diarization_speakers": conc,
                "has_untranscribed_overlap_risk": len(conc) >= 2,
                "zero_duration_words": sum(1 for w in cur_words if w["zero_duration"]),
            }
        )
        cur_words = []
        cur_spk = None

    for w in attributed_words:
        sp = w["speaker"]
        if cur_words and (sp != cur_spk or (w["start"] is not None and cur_words[-1]["end"] is not None and (w["start"] - cur_words[-1]["end"]) > 1.2)):
            flush()
        cur_spk = sp
        cur_words.append(w)
    flush()

    # Record diarization regions with multi-speaker overlap that single-stream ASR cannot separate
    overlap_diar_regions = []
    for i, a in enumerate(diar_segments):
        for b in diar_segments[i + 1 :]:
            if b["start"] >= a["end"]:
                break
            ov_s = min(a["end"], b["end"]) - max(a["start"], b["start"])
            if ov_s >= 0.35 and a["speaker"] != b["speaker"]:
                overlap_diar_regions.append(
                    {
                        "start": round(max(a["start"], b["start"]), 3),
                        "end": round(min(a["end"], b["end"]), 3),
                        "duration_s": round(ov_s, 3),
                        "speakers": [f"SPK_{a['speaker']:02d}", f"SPK_{b['speaker']:02d}"],
                    }
                )

    return {
        "schema_version": "experimental-modular-contract/v1",
        "variant": variant,
        "source": {"path": source_wav, "duration_s": duration_s, "offset_s": source_offset_s},
        "source_text_preserved": source_text,
        "stats": {
            **stats,
            "utterance_count": len(utterances),
            "speakers_in_utterances": sorted({u["speaker"] for u in utterances}),
            "diarization_speaker_count": len({s["speaker"] for s in diar_segments}),
            "overlap_diarization_region_count": len(overlap_diar_regions),
        },
        "overlap_diarization_regions": overlap_diar_regions[:30],
        "utterances": utterances,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-json", default=str(MODULAR_OUT / "raw/qwen_asr_eval.json"))
    parser.add_argument("--align-json", default=str(MODULAR_OUT / "raw/qwen_aligner_f300_chunked.json"))
    parser.add_argument("--diar-json", default=str(MODULAR_OUT / "raw/nemotron_diar_eval.json"))
    parser.add_argument("--out", default=str(MODULAR_OUT / "raw/integration_probe.json"))
    args = parser.parse_args()

    asr_data = json.loads(Path(args.asr_json).read_text(encoding="utf-8"))
    align_data = json.loads(Path(args.align_json).read_text(encoding="utf-8"))
    diar_data = json.loads(Path(args.diar_json).read_text(encoding="utf-8"))

    diar_segs = diar_data["cases"]["F_300s"]["streaming"]["segments"]
    source_wav = str(WORKSPACE_ROOT / "outputs/audio/fr/fr_0230_0730_300s.wav")

    probes: dict[str, Any] = {}
    for variant_key, align_entry in align_data["variants"].items():
        words = align_entry["items"]
        src_text = align_entry["source_text"]
        attr_words, stats = attribute_words_with_diarization(words, diar_segs)
        probes[variant_key] = build_experimental_contract(
            variant=variant_key,
            source_wav=source_wav,
            duration_s=300.0,
            source_offset_s=150.0,
            source_text=src_text,
            attributed_words=attr_words,
            diar_segments=diar_segs,
            stats=stats,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(probes, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {k: v["stats"] for k, v in probes.items()}
    print(json.dumps({"out": str(out_path), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
