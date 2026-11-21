#!/usr/bin/env python3
"""Read-only compatibility smoke test feeding normalized Nemotron samples into /workspace/pyannote's subtitle writer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/workspace/nvidia-transcribe")
PYANNOTE_REPO = Path("/workspace/pyannote")


def main() -> int:
    if not PYANNOTE_REPO.is_dir():
        print("Reference repo /workspace/pyannote not found", file=sys.stderr)
        return 1

    sys.path.insert(0, str(PYANNOTE_REPO))
    from pyannote_app.utils.subtitles import build_subtitles  # noqa: E402

    from nvidia_transcribe.contract import to_pyannote_compatible_segments  # noqa: E402

    renders_dir = ROOT / "outputs/renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    # 1. French multi-speaker sample (39s child voices excerpt, absolute timestamps)
    fr_contract = json.loads(
        (ROOT / "outputs/responses/fr/fr_0316_0355_child_voices_speaker.contract.json").read_text(encoding="utf-8")
    )
    fr_segs = to_pyannote_compatible_segments(fr_contract)
    fr_srt, fr_ass, fr_overflow = build_subtitles(
        fr_segs,
        renders_dir / "fr_0316_0355_nemotron.srt",
        renders_dir / "fr_0316_0355_nemotron.ass",
    )

    # 2. Mandarin Chinese sample (preserving CJK text without inter-character spaces)
    zh_files = sorted((ROOT / "outputs/responses/zh").glob("*_auto.contract.json"))
    zh_report = {}
    if zh_files:
        zh_contract = json.loads(zh_files[0].read_text(encoding="utf-8"))
        zh_segs = to_pyannote_compatible_segments(zh_contract)
        zh_utts = [
            {
                "id": u["id"],
                "source_language": "zh",
                "speaker": u["speaker_id"],
                "source_text": u["source_text"],
                "start": u["source_start"],
                "end": u["source_end"],
                "timing_status": "word-aligned",
                "word_count": len(u.get("source_characters") or []) + len(u.get("source_words") or []),
                "provenance": {"segment_index": idx},
                "source_words": [
                    {
                        "word": c["text"],
                        "start": c["start"],
                        "end": c["end"],
                        "speaker": c["speaker"],
                    }
                    for c in (u.get("source_characters") or []) + (u.get("source_words") or [])
                    if c.get("valid_timing")
                ],
            }
            for idx, u in enumerate(zh_contract.get("utterances", []))
            if u.get("source_start") is not None and u.get("source_end") is not None
        ]
        zh_srt, zh_ass, zh_overflow = build_subtitles(
            {"segments": zh_segs, "subtitle_utterances": zh_utts},
            renders_dir / "zh_sample_nemotron.srt",
            renders_dir / "zh_sample_nemotron.ass",
        )
        zh_report = {
            "srt": str(zh_srt),
            "ass": str(zh_ass),
            "overflow_count": len(zh_overflow),
            "srt_preview": zh_srt.read_text(encoding="utf-8")[:600],
        }

    summary = {
        "fr": {
            "srt": str(fr_srt),
            "ass": str(fr_ass),
            "segment_count": len(fr_segs),
            "overflow_count": len(fr_overflow),
            "srt_preview": fr_srt.read_text(encoding="utf-8")[:600],
        },
        "zh": zh_report,
    }
    (renders_dir / "compat_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
