"""CPU tests for standalone English SRT/ASS subtitle compositor (Work Package D)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nvidia_transcribe.subtitles import (
    SubtitleConfig,
    assign_subtitle_lanes,
    build_display_cues,
    format_ass,
    format_srt,
    render_subtitles,
    wrap_subtitle_lines,
)


def test_wrap_subtitle_lines_prefers_phrase_breaks_and_balance() -> None:
    text = "I told you earlier, we have to leave before sundown."
    lines = wrap_subtitle_lines(text, max_chars=42, max_lines=2)
    assert len(lines) == 2
    assert lines[0] == "I told you earlier,"
    assert lines[1] == "we have to leave before sundown."
    assert all(len(ln) <= 42 for ln in lines)


def test_build_display_cues_adjusts_duration_and_keeps_unrenderable_honest() -> None:
    utts = [
        {"id": "utt-0000", "speaker_id": "SPK_00", "source_start": 1.0, "source_end": 1.4, "translated_text": "Look out!"},
        {"id": "utt-0001", "speaker_id": "SPK_00", "source_start": 5.0, "source_end": 15.0, "translated_text": "Too long without split check."},
        {"id": "utt-0002", "speaker_id": "SPK_01", "source_start": None, "source_end": None, "translated_text": "No timing evidence here."},
        {"id": "utt-0003", "speaker_id": "SPK_00", "source_start": 16.0, "source_end": 18.0, "translated_text": "Partial timing line.", "timestamp_provenance": {"timing_status": "partial"}},
    ]
    cues = build_display_cues(utts, config=SubtitleConfig(min_display_s=1.2, max_display_s=6.5))
    assert len(cues) == 4
    # Short cue extended to min_display_s while raw_end keeps original 1.4
    assert cues[0].start == 1.0
    assert cues[0].end == 2.2
    assert cues[0].raw_end == 1.4
    # Long cue clamped to max_display_s without losing text
    assert cues[1].end - cues[1].start == pytest.approx(6.5)
    assert "max_display_clamped" in cues[1].conflicts
    # Missing timing stays unrenderable (no fabricated 0.0s cue in SRT/ASS)
    assert cues[2].unrenderable is True
    # Partial timing still renders with provenance warning
    assert cues[3].unrenderable is False
    assert "source_timing_partial" in cues[3].conflicts


def test_real_vs_display_induced_overlap_and_ass_lanes() -> None:
    # Two sequential speakers whose short cues get padded to min_display_s=1.2s;
    # acoustic intervals [1.0..1.4] and [1.5..1.9] do NOT overlap!
    utts = [
        {"id": "utt-0", "speaker_id": "SPK_00", "source_start": 1.0, "source_end": 1.4, "translated_text": "First speaker."},
        {"id": "utt-1", "speaker_id": "SPK_01", "source_start": 1.5, "source_end": 1.9, "translated_text": "Second speaker."},
    ]
    cues = build_display_cues(utts, config=SubtitleConfig(min_display_s=1.2))
    displayed, overflow, _ = assign_subtitle_lanes(cues, utterances=utts, overlap_threshold_s=0.4)
    assert overflow == []
    # Display padding must NEVER trigger false simultaneous Top lane!
    assert all(c.lane == 0 for c in displayed)
    assert displayed[0].end <= displayed[1].start

    # Genuine acoustic overlap >= 0.4s MUST separate onto Main (0) and Top (1)
    ov_utts = [
        {"id": "utt-0", "speaker_id": "SPK_00", "source_start": 10.0, "source_end": 13.0, "translated_text": "Main active speaker."},
        {"id": "utt-1", "speaker_id": "SPK_01", "source_start": 10.4, "source_end": 12.2, "translated_text": "Secondary interrupting speaker."},
        {"id": "utt-2", "speaker_id": "SPK_02", "source_start": 10.6, "source_end": 12.0, "translated_text": "Third simultaneous voice."},
    ]
    ov_cues = build_display_cues(ov_utts)
    disp2, ov2, _ = assign_subtitle_lanes(ov_cues, utterances=ov_utts, overlap_threshold_s=0.4)
    lanes_by_id = {c.utterance_id: c.lane for c in disp2}
    assert lanes_by_id["utt-0"] == 0
    assert lanes_by_id["utt-1"] == 1
    assert [c.utterance_id for c in ov2] == ["utt-2"]

    # SRT retains ALL dialogue including 3rd-speaker overflow; ASS has valid styles & Main/Top events
    srt_text = format_srt(disp2 + ov2)
    ass_text = format_ass(disp2, title="Overlap Test")
    assert "Main active speaker." in srt_text
    assert "Secondary interrupting speaker." in srt_text
    assert "Third simultaneous voice." in srt_text
    assert "[Script Info]" in ass_text and "ScriptType: v4.00+" in ass_text
    assert ",Main," in ass_text and ",Top," in ass_text


def test_render_subtitles_and_safe_sidecar_overwrite(tmp_path: Path) -> None:
    contract = {
        "utterances": [
            {"id": "utt-0000", "speaker_id": "SPK_00", "source_start": 1.0, "source_end": 3.0, "translated_text": "Let's go, Tao!"}
        ]
    }
    out_dir = tmp_path / "run"
    side_dir = tmp_path / "media"
    side_dir.mkdir()
    existing_srt = side_dir / "Episode01.en.srt"
    existing_srt.write_text("existing", encoding="utf-8")

    # Refuse silent overwrite when sidecar already exists
    with pytest.raises(FileExistsError, match="overwrite_sidecar"):
        render_subtitles(
            contract,
            output_dir=out_dir,
            episode_stem="Episode01",
            sidecar_dir=side_dir,
            overwrite_sidecar=False,
        )

    res = render_subtitles(
        contract,
        output_dir=out_dir,
        episode_stem="Episode01",
        sidecar_dir=side_dir,
        overwrite_sidecar=True,
    )
    assert (out_dir / "Episode01.en.srt").is_file()
    assert (out_dir / "Episode01.en.ass").is_file()
    assert "Let's go, Tao!" in existing_srt.read_text(encoding="utf-8")
    assert res["stats"]["renderable_cues"] == 1
