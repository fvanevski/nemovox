"""Focused CPU regression tests for Viewing-Led Subtitle Validation (overlay recovery & export)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nvidia_transcribe.cli import build_parser
from nvidia_transcribe.subtitle_pipeline import SubtitlePipelineConfig, generate_subtitles
from nvidia_transcribe.subtitles import SubtitleConfig, render_subtitles
from nvidia_transcribe.utterances import apply_validation_overlay


def test_overlay_recovers_coarse_timing_preserves_ids_and_prevents_duplicates() -> None:
    contract = {
        "source_language": "zh",
        "utterances": [
            {
                "id": "utt-0102",
                "original_utterance_id": "utt-0259",
                "speaker_id": "SPK_15",
                "source_start": None,
                "source_end": None,
                "source_text": "今夜之后，新任冥皇登基",
                "translated_text": "After tonight, the new Underworld Emperor ascends.",
                "timestamp_provenance": {"timing_status": "failed"},
            },
            {
                "id": "utt-0205",
                "original_utterance_id": "utt-0269",
                "speaker_id": "SPK_13",
                "source_start": None,
                "source_end": None,
                "source_text": "天呐。",
                "translated_text": "Good heavens!",
                "timestamp_provenance": {"timing_status": "failed"},
            },
            {
                "id": "utt-0223",
                "original_utterance_id": "utt-0273",
                "speaker_id": "SPK_13",
                "source_start": None,
                "source_end": None,
                "source_text": "天哪！",
                "translated_text": "Oh my god!",
                "timestamp_provenance": {"timing_status": "failed"},
            },
            {
                "id": "utt-0243",
                "speaker_id": "SPK_18",
                "source_start": 1064.88,
                "source_end": 1065.46,
                "source_text": "这所有的一切，都是我用命换来的。",
                "translated_text": "All bought with my life.",
                "timestamp_provenance": {"timing_status": "partial"},
            },
        ],
    }
    overlay = {
        "name": "test_zh",
        "utterances": [
            {
                "utterance_id": "utt-0102",
                "original_utterance_id": "utt-0259",
                "expected_source_substring": "新任冥皇",
                "proposed_start_s": 633.1,
                "proposed_end_s": 641.5,
            },
            {
                "utterance_id": "utt-0205",
                "expected_source_substring": "天呐",
                "suppressed_duplicate_of": "utt-0223",
                "reason": "overlap_dup",
            },
            {
                "utterance_id": "utt-0223",
                "original_utterance_id": "utt-0273",
                "expected_source_substring": "天哪",
                "proposed_start_s": 985.2,
                "proposed_end_s": 986.3,
            },
            {
                "utterance_id": "utt-0243",
                "expected_source_substring": "用命换来的",
                "proposed_start_s": 1071.9,
                "proposed_end_s": 1078.8,
                "translated_text": "Everything I have, I earned with my life!",
            },
        ],
    }
    updated, summary = apply_validation_overlay(contract, overlay)
    assert summary["recovered_timing_count"] == 2
    assert summary["relocated_timing_count"] == 1
    assert summary["edited_text_count"] == 1
    assert summary["suppressed_count"] == 1

    by_id = {u["id"]: u for u in updated["utterances"]}
    # Original/reconstructed IDs preserved without shifting
    assert set(by_id.keys()) == {"utt-0102", "utt-0205", "utt-0223", "utt-0243"}
    assert by_id["utt-0102"]["original_utterance_id"] == "utt-0259"
    assert by_id["utt-0102"]["source_start"] == 633.1 and by_id["utt-0102"]["source_end"] == 641.5
    assert by_id["utt-0102"]["timing_status"] == "coarse_recovered"
    assert by_id["utt-0205"]["suppressed_duplicate_of"] == "utt-0223"
    assert by_id["utt-0205"]["translated_text"] == ""
    # Relocated item keeps raw ASR provenance and prior wording
    assert by_id["utt-0243"]["raw_source_start"] == 1064.88
    assert by_id["utt-0243"]["pre_overlay_translated_text"] == "All bought with my life."

    # Mismatch guard prevents silent wrong-ID edit
    with pytest.raises(ValueError, match="mismatch"):
        apply_validation_overlay(
            contract,
            {"utterances": [{"utterance_id": "utt-0102", "expected_source_substring": "DOES_NOT_EXIST"}]},
        )


def test_cli_and_rerender_with_validation_overlay_export(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["generate-subtitles", "--recording-contract", "r.json", "--output-dir", str(tmp_path), "--re-render", "--validation-overlay", "ov.json"]
    )
    assert args.validation_overlay == "ov.json"

    translated = {
        "source_language": "zh",
        "target_language": "en",
        "source": {"path": "/tmp/ep166.webm", "duration_s": 1200.0},
        "utterances": [
            {
                "id": "utt-0010",
                "speaker_id": "SPK_01",
                "source_start": 100.0,
                "source_end": 103.0,
                "source_text": "一。",
                "translated_text": "First line.",
                "translation_status": "translated",
            },
            {
                "id": "utt-0011",
                "speaker_id": "SPK_02",
                "source_start": None,
                "source_end": None,
                "source_text": "二。",
                "translated_text": "Recovered second line.",
                "translation_status": "translated",
            },
            {
                "id": "utt-0012",
                "speaker_id": "SPK_01",
                "source_start": 101.0,
                "source_end": 101.2,
                "source_text": "三",
                "translated_text": "Noise",
                "translation_status": "translated",
            },
        ],
    }
    ov_path = tmp_path / "overlay.json"
    ov_path.write_text(
        json.dumps(
            {
                "name": "cli_test",
                "utterances": [
                    {"utterance_id": "utt-0011", "proposed_start_s": 104.0, "proposed_end_s": 107.0},
                    {"utterance_id": "utt-0012", "suppress": True, "reason": "noise"},
                ],
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "translated.contract.json").write_text(json.dumps(translated), encoding="utf-8")

    cfg = SubtitlePipelineConfig(
        re_render_only=True,
        validation_overlay_path=ov_path,
        subtitles=SubtitleConfig(ass_layout="raised_main", raised_main_margin_v=150),
    )
    res = generate_subtitles(output_dir=out_dir, config=cfg)
    assert res["validation_overlay"]["recovered_timing_count"] == 1
    assert res["validation_overlay"]["suppressed_count"] == 1
    assert res["subtitles"]["renderable_cues"] == 2
    assert res["subtitles"]["ass_displayed_cues"] == 2
    assert res["subtitles"]["ass_overflow_cues"] == 0

    srt_txt = Path(res["files"]["srt"]).read_text(encoding="utf-8")
    ass_txt = Path(res["files"]["ass"]).read_text(encoding="utf-8")
    assert "First line." in srt_txt and "Recovered second line." in srt_txt
    assert "Noise" not in srt_txt and "Noise" not in ass_txt
    assert "2,54,54,150,1" in ass_txt
