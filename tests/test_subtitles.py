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
    # Discretionary ASS stacking protects a short turn's reading time.
    assert {c.lane for c in displayed} == {0, 1}
    assert displayed[0].end > displayed[1].start
    assert "sequential_display_overlap" in displayed[1].conflicts
    # SRT does not inherit discretionary stacking.
    srt, _, _ = assign_subtitle_lanes(cues, utterances=utts, allow_sequential_stacking=False)
    assert all(c.lane == 0 for c in srt)
    assert srt[0].end <= srt[1].start

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


def test_short_distinct_voice_overlap_preserves_main_readability(tmp_path):
    line = "The courier delivered the supplies to the shelter before night."
    rows = [
        {"id": "a", "speaker_id": "SPK_A", "source_start": 10.0,
         "source_end": 12.82, "translated_text": line},
        {"id": "b", "speaker_id": "SPK_B", "source_start": 10.4,
         "source_end": 10.615, "translated_text": "Yes."},
    ]
    displayed, overflow, _ = assign_subtitle_lanes(build_display_cues(rows), utterances=rows)
    assert not overflow
    by_id = {c.utterance_id: c for c in displayed}
    assert by_id["a"].lane != by_id["b"].lane
    assert by_id["a"].end - by_id["a"].start >= len(line) / 25
    result = render_subtitles({"utterances": rows}, output_dir=tmp_path, episode_stem="test")
    assert result["stats"]["ass_displayed_cues"] == 2
    assert ",Top," in Path(result["outputs"]["ass"]).read_text()
    assert not any("display_clamped_before_next_cue" in c["conflicts"]
                   for c in result["conflicts"] if c["utterance_id"] == "a")


@pytest.mark.parametrize("duration", [72.0, 72.004])
def test_media_end_bounds_both_formats_and_keeps_unrenderable_visible(tmp_path, duration):
    rows = [
        {"id": "last", "speaker_id": "SPK_A", "source_start": 71.9,
         "source_end": 72., "translated_text": "Yes."},
        {"id": "outside", "speaker_id": "SPK_B", "source_start": 72.1,
         "source_end": 72.3, "translated_text": "Outside."},
    ]
    res = render_subtitles({"source": {"duration_s": duration}, "utterances": rows},
                           output_dir=tmp_path, episode_stem="end")
    assert "00:01:11,900 --> 00:01:12,000" in Path(res["outputs"]["srt"]).read_text()
    assert "0:01:11.90,0:01:12.00" in Path(res["outputs"]["ass"]).read_text()
    assert res["stats"]["unrenderable_cues"] == 1
    assert "Outside." not in Path(res["outputs"]["srt"]).read_text()
    assert any("unrenderable_outside_media_duration" in c["conflicts"] for c in res["conflicts"])


def test_all_unlocalized_dialogue_does_not_publish_empty_sidecars(tmp_path):
    sidecars=tmp_path/"sidecars"
    sidecars.mkdir()
    res=render_subtitles({"utterances":[{"id":"x","source_text":"Known speech",
        "translated_text":"Known speech","source_start":None,"source_end":None}]},
        output_dir=tmp_path,episode_stem="untimed",sidecar_dir=sidecars)
    assert res["no_renderable_dialogue"] is True
    assert res["stats"]["unrenderable_cues"]==1
    assert not res["sidecars"] and not list(sidecars.iterdir())
    assert Path(res["outputs"]["srt"]).read_bytes()==b""
    silent=render_subtitles({"utterances":[]},output_dir=tmp_path,episode_stem="silence")
    assert silent["no_renderable_dialogue"] is False


def test_saturated_ass_retains_explicit_lineage_and_srt(tmp_path):
    rows = [{"id": str(i), "speaker_id": f"SPK_{i}", "source_start": 1.0,
             "source_end": 3.0, "translated_text": f"Distinct voice {i}."}
            for i in range(4)]
    result = render_subtitles({"utterances": rows}, output_dir=tmp_path,
                              episode_stem="saturation", config=SubtitleConfig(ass_max_lanes=2))
    assert result["stats"]["ass_displayed_cues"] == 2
    assert result["stats"]["ass_overflow_cues"] == 2
    assert {r["occurrence_id"] for r in result["cue_lineage"]} == {"0", "1", "2", "3"}
    assert all(r["display_interval"] is None for r in result["cue_lineage"]
               if r["disposition"] == "overflow")
    assert all(r["disposition"] == "displayed" for r in result["srt_cue_lineage"])
    assert all(f"Distinct voice {i}." in Path(result["outputs"]["srt"]).read_text()
               for i in range(4))


def test_split_allocations_do_not_extend_source_interval():
    row = {"id": "tiny", "source_start": 2.0, "source_end": 2.08,
           "translated_text": " ".join(["A different phrase follows."] * 8)}
    cues = build_display_cues([row], config=SubtitleConfig(max_chars=30))
    assert len(cues) > 1
    assert cues[0].raw_start == 2.0
    assert cues[-1].raw_end == 2.08
    assert all(2.0 <= c.raw_start < c.raw_end <= 2.08 for c in cues)
    assert all(a.raw_end == b.raw_start for a, b in zip(cues, cues[1:]))
    assert " ".join(c.text for c in cues) == row["translated_text"]


def test_discretionary_stacking_declined_under_combined_pressure():
    rows = [
        {"id": "a", "speaker_id": "SPK_A", "source_start": 1.0,
         "source_end": 1.4, "translated_text": "The entire mountain is covered in mist and the route ahead is unclear."},
        {"id": "b", "speaker_id": "SPK_B", "source_start": 1.5,
         "source_end": 1.9, "translated_text": "We should return to the village before the dangerous weather arrives."},
    ]
    placed, overflow, missing = assign_subtitle_lanes(build_display_cues(rows))
    assert not overflow and not missing
    assert all(c.lane == 0 for c in placed)
    assert placed[0].end <= placed[1].start
    assert any(c.startswith("high_reading_speed:") for c in placed[0].conflicts)


def test_same_speaker_concurrent_distinct_occurrences_not_deleted():
    rows = [{"id": str(i), "speaker_id": "SPK_A", "source_start": 1,
             "source_end": 2, "translated_text": "Yes."} for i in range(2)]
    placed, overflow, missing = assign_subtitle_lanes(build_display_cues(rows))
    assert not overflow and not missing
    assert {c.utterance_id for c in placed} == {"0", "1"}
    assert {c.lane for c in placed} == {0, 1}


def test_joint_continuation_groups_only_with_membership_and_no_intervening_event():
    selected = {"unit_id": "joint", "selection": "jointly_adjudicated",
                "constituents": [{"id": "a"}, {"id": "b"}]}
    a = {"id": "a", "speaker_id": "SPK_A", "source_start": 1.0,
         "source_end": 1.5, "translated_text": "We must", "translation_unit": selected}
    b = {"id": "b", "speaker_id": "SPK_A", "source_start": 1.55,
         "source_end": 2.1, "translated_text": "leave now.", "translation_unit": selected}
    grouped = build_display_cues([a, b])
    assert len(grouped) == 1 and grouped[0].text == "We must leave now."
    assert grouped[0].display_occurrence_ids == ["a", "b"]
    assert grouped[0].source_intervals == [(1.0, 1.5), (1.55, 2.1)]
    assert len(build_display_cues([a, {**b, "translation_unit": None}])) == 2
    assert len(build_display_cues([a, {**b, "translated_text": "Leave now."}])) == 2
    assert len(build_display_cues([a, {**b, "speaker_id": "SPK_B"}])) == 2
    other = {"id": "c", "speaker_id": "SPK_B", "source_start": 1.2,
             "source_end": 1.8, "translated_text": "Wait."}
    assert len(build_display_cues([a, b, other])) == 3
