"""CPU unit tests for the modular multilingual transcription evaluation suite."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.modular.common import (
    compute_cer_detailed,
    compute_overlap_der,
    compute_wer_detailed,
    normalize_language,
    offset_intervals,
    preserve_unicode_text,
    validate_alignment_items,
)
from experiments.modular.integration.run_integration_probe import (
    attribute_words_with_diarization,
    build_experimental_contract,
)


def test_source_language_normalization_never_defaults_to_french() -> None:
    assert normalize_language("French") == "fr"
    assert normalize_language("Mandarin Chinese") == "zh"
    assert normalize_language("Chinese, Mandarin") == "zh"
    assert normalize_language("Cantonese") == "yue"
    assert normalize_language(None) == "unknown"
    assert normalize_language("  ") == "unknown"


def test_french_and_chinese_unicode_preservation() -> None:
    fr = "Les mystérieuses cités d'or — « on ne devrait pas »"
    zh = "钙、钾等元素属于金属，银和金等元素当然也是金属。"
    assert preserve_unicode_text(fr) == fr
    assert preserve_unicode_text(zh) == zh
    assert preserve_unicode_text(None) == ""


def test_alignment_validation_detects_zero_duration_inversions_and_bounds() -> None:
    expected = "Hello world test"
    items = [
        {"text": "Hello", "start": 0.1, "end": 0.5},
        {"text": "world", "start": 0.5, "end": 0.5},  # zero duration (Issue #197)
        {"text": "test", "start": 2.4, "end": 2.1},   # inverted + non-monotonic? monotonic ok, inverted!
    ]
    report = validate_alignment_items(items, expected_text=expected, audio_duration_s=2.0)
    assert report["zero_duration_count"] == 1
    assert report["negative_or_inverted_count"] == 1
    assert report["past_duration_count"] == 1
    assert report["valid_timed_items"] == 1
    assert report["text_preserved_exact"] is True


def test_alignment_validation_detects_repeated_spans_and_missing_chars() -> None:
    expected = "钙钾等金属"
    items = [
        {"text": "钙", "start": 0.2, "end": 0.6},
        {"text": "等", "start": 0.2, "end": 0.6},  # repeated identical span + missing '钾'
        {"text": "金", "start": 0.9, "end": 0.9},  # zero-duration
        {"text": "属", "start": None, "end": None},  # missing timing preserved as null
    ]
    report = validate_alignment_items(items, expected_text=expected, audio_duration_s=2.0)
    assert report["repeated_identical_span_count"] == 1
    assert report["zero_duration_count"] == 1
    assert report["missing_timing_count"] == 1
    assert report["text_preserved_exact"] is False
    assert report["lexical_missing_chars_approx"] == 1


def test_global_timestamp_offsets_and_unknown_speaker_preserved() -> None:
    ivs = [
        {"speaker": None, "start": 1.25, "end": 2.5},
        {"speaker": "", "start": None, "end": None},
        {"speaker": "speaker_1", "start": 2.4, "end": 3.8},
    ]
    shifted = offset_intervals(ivs, source_offset_s=196.0)
    assert shifted[0]["speaker"] == "unknown"
    assert shifted[0]["start"] == 197.25
    assert shifted[0]["end"] == 198.5
    # Null timing must remain null, never fabricated
    assert shifted[1]["speaker"] == "unknown"
    assert shifted[1]["start"] is None
    assert shifted[1]["end"] is None
    assert shifted[2]["speaker"] == "speaker_1"
    assert shifted[2]["start"] == 198.4


def test_overlap_der_and_attributed_overlap_metrics() -> None:
    ref = [(0.0, 4.0, "A"), (2.0, 5.0, "B")]
    # Perfect match with swapped labels (speaker_1->A, speaker_0->B)
    hyp_good = [(0.0, 4.0, "speaker_1"), (2.0, 5.0, "speaker_0")]
    m_good = compute_overlap_der(ref, hyp_good, total_duration_s=5.0)
    assert m_good["der"] == 0.0
    assert m_good["jer"] == 0.0
    assert m_good["overlap_f1"] == 1.0

    # Collapsed secondary speaker -> misses 3s of B overlap
    hyp_collapsed = [(0.0, 5.0, "speaker_0")]
    m_bad = compute_overlap_der(ref, hyp_collapsed, total_duration_s=5.0)
    assert m_bad["missed_s"] >= 2.0
    assert m_bad["overlap_recall"] == 0.0


def test_non_destructive_wer_cer_and_json_roundtrip(tmp_path: Path) -> None:
    ref_zh = "钙、钾等元素属于金属，银和金等元素当然也是金属。"
    hyp_zh = "钙、钾等元素属于金属，银和金等元素当然也是金属。"
    cer = compute_cer_detailed(ref_zh, hyp_zh)
    assert cer["cer"] == 0.0
    assert cer["edits"] == 0

    wer = compute_wer_detailed("C'est même comme ça !", "C'est comme ça")
    assert wer["deletions"] == 1
    assert wer["substitutions"] == 0

    payload = {"source_text": ref_zh, "raw_outputs": {"align": [{"text": "钾", "start": None, "end": None}]}}
    out_file = tmp_path / "artifact.json"
    out_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded == payload


def test_integration_attribution_invariants() -> None:
    diar = [
        {"speaker": 0, "start": 10.0, "end": 14.0},
        {"speaker": 1, "start": 12.0, "end": 16.0},
    ]
    words = [
        {"text": "Bonjour", "start": 10.2, "end": 10.8},
        {"text": "tout", "start": 12.5, "end": 12.5},  # zero-duration equal overlap -> concurrent kept, speaker=unknown
        {"text": "silence", "start": 20.0, "end": 20.5},  # no diarization -> must stay unknown
        {"text": "manquant", "start": None, "end": None},  # unaligned -> must not be dropped
    ]
    attr, stats = attribute_words_with_diarization(words, diar)
    assert [w["text"] for w in attr] == ["Bonjour", "tout", "silence", "manquant"]
    assert attr[0]["speaker"] == "SPK_00"
    assert sorted(attr[1]["concurrent_speakers"]) == ["SPK_00", "SPK_01"]
    assert attr[1]["speaker"] == "unknown"
    assert attr[2]["speaker"] == "unknown"
    assert attr[3]["speaker"] == "unknown"
    assert attr[3]["unaligned"] is True
    assert stats["zero_duration_words"] == 1
    assert stats["overlap_flagged_words"] == 1
    assert stats["unknown_words"] == 3

    contract = build_experimental_contract(
        variant="unit_probe",
        source_wav="dummy.wav",
        duration_s=25.0,
        source_offset_s=0.0,
        source_text="Bonjour tout silence manquant",
        attributed_words=attr,
        diar_segments=diar,
        stats=stats,
    )
    assert contract["stats"]["overlap_diarization_region_count"] == 1
    assert contract["source_text_preserved"] == "Bonjour tout silence manquant"

