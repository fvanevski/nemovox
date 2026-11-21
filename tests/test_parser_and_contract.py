"""CPU-only unit tests for speaker tags, CTC timestamps, Unicode, offsets, and contract."""

from __future__ import annotations

import json

import pytest

from nvidia_transcribe.contract import build_transcript_contract, to_pyannote_compatible_segments
from nvidia_transcribe.parser import (
    CTCMetadataError,
    parse_ctc_timestamps,
    parse_speaker_turns,
    split_lexical_units,
)


def test_speaker_tag_extraction_and_events() -> None:
    raw = "<spk:0> Bonjour, ça va ? <spk:1> Oui ! <ev:laughter> Très bien."
    turns = parse_speaker_turns(raw)
    assert len(turns) == 2
    assert turns[0].speaker_tag == "<spk:0>"
    assert turns[0].text == "Bonjour, ça va ?"
    assert turns[1].speaker_tag == "<spk:1>"
    assert turns[1].events == ("laughter",)
    assert turns[1].text == "Oui ! Très bien."


def test_french_unicode_and_source_offset_conversion() -> None:
    ctc = {
        "words": [
            {"word": "C'était", "start": 0.25, "end": 0.75, "speaker": "<spk:0>"},
            {"word": "l'été", "start": 0.80, "end": 1.30, "speaker": "<spk:0>"},
            {"word": "à", "start": 1.35, "end": 1.50, "speaker": "<spk:1>"},
            {"word": "Château", "start": 1.55, "end": 2.10, "speaker": "<spk:1>"},
        ],
        "diarization": [
            {"speaker": 0, "start": 0.20, "end": 1.32},
            {"speaker": 1, "start": 1.34, "end": 2.15},
        ],
        "speaker_tag_to_diarization_speaker": {"<spk:0>": 0, "<spk:1>": 1},
        "error": None,
    }
    contract = build_transcript_contract(
        raw_text="<spk:0> C'était l'été <spk:1> à Château",
        source_language="fr",
        ctc_payload=ctc,
        source_offset_s=196.0,
        model_provenance={"model": "nemotron-3.5-transcribe-nvfp4"},
    )
    assert contract["source_language"] == "fr"
    assert contract["source_text"] == "C'était l'été à Château"
    assert contract["uncertainty"] is None
    assert contract["speaker_mapping"] == {"<spk:0>": 0, "<spk:1>": 1}

    utts = contract["utterances"]
    assert len(utts) == 2
    assert utts[0]["speaker_id"] == "<spk:0>"
    assert utts[0]["source_start"] == 196.25
    assert utts[0]["source_end"] == 197.30
    assert utts[0]["source_words"][0]["raw_start"] == 0.25
    assert utts[1]["source_start"] == 197.35
    assert contract["diarization_intervals"][1]["start"] == 197.34

    segs = to_pyannote_compatible_segments(contract)
    assert len(segs) == 2
    assert segs[0]["words"][0]["word"] == "C'était"
    assert segs[0]["words"][0]["start"] == 196.25


def test_chinese_unicode_and_character_timing() -> None:
    units = split_lexical_units("你好世界 hello")
    assert [u["text"] for u in units] == ["你", "好", "世", "界", "hello"]
    assert [u["unit_type"] for u in units] == ["character", "character", "character", "character", "word"]

    ctc = {
        "words": [
            {"word": "你", "start": 0.10, "end": 0.22, "speaker": "<spk:0>"},
            {"word": "好", "start": 0.22, "end": 0.40, "speaker": "<spk:0>"},
            {"word": "世", "start": 0.42, "end": 0.60, "speaker": "<spk:0>"},
            {"word": "界", "start": 0.60, "end": 0.85, "speaker": "<spk:0>"},
        ],
        "diarization": [{"speaker": 0, "start": 0.08, "end": 0.90}],
        "speaker_tag_to_diarization_speaker": {"<spk:0>": 0},
        "error": None,
    }
    contract = build_transcript_contract(
        raw_text="<spk:0> 你好世界",
        source_language="zh",
        ctc_payload=ctc,
        source_offset_s=10.0,
    )
    utt = contract["utterances"][0]
    assert utt["source_text"] == "你好世界"
    assert len(utt["source_characters"]) == 4
    assert utt["source_words"] == []
    assert utt["source_characters"][0]["start"] == 10.10


def test_missing_and_invalid_timestamps_never_fabricated() -> None:
    bad_ctc = {
        "words": [
            {"word": "ok", "start": 0.1, "end": 0.3, "speaker": "<spk:0>"},
            {"word": "nan_word", "start": float("nan"), "end": 0.5, "speaker": "<spk:0>"},
            {"word": "neg_word", "start": -0.2, "end": 0.8, "speaker": "<spk:0>"},
            {"word": "inv_word", "start": 1.2, "end": 0.9, "speaker": "<spk:0>"},
        ],
        "diarization": [],
        "speaker_tag_to_diarization_speaker": {},
        "error": None,
    }
    parsed = parse_ctc_timestamps(bad_ctc)
    assert parsed.invalid_timing_count == 3
    assert parsed.units[0].valid_timing is True
    for bad in parsed.units[1:]:
        assert bad.valid_timing is False
        assert bad.start is None
        assert bad.end is None

    with pytest.raises(CTCMetadataError):
        parse_ctc_timestamps(None, strict=True)
    with pytest.raises(CTCMetadataError):
        parse_ctc_timestamps({"error": "alignment_failed", "words": []}, strict=True)


def test_source_transcript_preserved_even_with_translation() -> None:
    contract = build_transcript_contract(
        raw_text="<spk:0> Bonjour tout le monde",
        source_language="fr",
        ctc_payload=None,
        english_translation="Hello everyone",
    )
    assert contract["source_text"] == "Bonjour tout le monde"
    assert contract["english_translation"] == "Hello everyone"
    assert contract["utterances"][0]["source_text"] == "Bonjour tout le monde"
    serialized = json.dumps(contract, ensure_ascii=False)
    assert "Bonjour tout le monde" in serialized
    assert "Hello everyone" in serialized
