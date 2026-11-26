"""CPU-only regression tests for overlap reconstruction, CTC recovery, and speaker reconciliation."""

from __future__ import annotations

import json
from pathlib import Path

from nvidia_transcribe.alignment import find_ctc_oov_characters, recover_chunk_alignment
from nvidia_transcribe.chunking import ChunkConfig, plan_chunks
from nvidia_transcribe.contract import build_transcript_contract
from nvidia_transcribe.speakers import normalize_chunk_speakers, reconcile_speakers
from nvidia_transcribe.utterances import reconstruct_speaker_utterances


def test_mandarin_interleaved_overlap_reconstructs_concurrent_streams() -> None:
    archived = Path("/workspace/nvidia-transcribe/outputs/responses/overlap/zh_2spk_overlap_sot.raw.json")
    raw = json.loads(archived.read_text(encoding="utf-8"))
    raw_text = raw["choices"][0]["message"]["content"]
    contract = build_transcript_contract(
        raw_text=raw_text,
        source_language="zh",
        ctc_payload=raw["ctc_timestamps"],
        source_offset_s=0.0,
    )
    utts = contract["utterances"]
    # Previously fragmented into 32 1-char utterances; now cleanly reconstructed per speaker!
    assert len(utts) <= 6
    spk0_text = "".join(u["source_text"] for u in utts if u["speaker_id"] == "0")
    spk1_text = "".join(u["source_text"] for u in utts if u["speaker_id"] == "1")
    assert "这并不是告别" in spk0_text
    assert "而是一个篇章的结束也是新篇章的开始" in spk0_text
    assert "斯科特四零三路公交车车" in spk1_text
    assert "停其发车前往新加兰在罗卡角停靠" in spk1_text
    # No inter-character spaces injected into Mandarin text
    assert "这 并 不 是" not in spk0_text
    # Overlapping time windows are preserved
    u0 = [u for u in utts if u["speaker_id"] == "0"][0]
    u1 = [u for u in utts if u["speaker_id"] == "1"][0]
    assert u0["source_start"] < u1["source_end"] and u1["source_start"] < u0["source_end"]


def test_french_overlap_and_pause_splitting() -> None:
    units = [
        {"text": "Bonjour", "start": 0.2, "end": 0.6, "speaker": "0", "unit_type": "word", "valid_timing": True},
        {"text": "Marie.", "start": 0.65, "end": 1.0, "speaker": "0", "unit_type": "word", "valid_timing": True},
        # Simultaneous speaker 1
        {"text": "Salut", "start": 0.5, "end": 0.8, "speaker": "1", "unit_type": "word", "valid_timing": True},
        {"text": "Paul!", "start": 0.85, "end": 1.2, "speaker": "1", "unit_type": "word", "valid_timing": True},
        # Meaningful pause (> 1.5s) for speaker 0 splits into second utterance
        {"text": "Comment", "start": 3.0, "end": 3.3, "speaker": "0", "unit_type": "word", "valid_timing": True},
        {"text": "vas-tu?", "start": 3.35, "end": 3.8, "speaker": "0", "unit_type": "word", "valid_timing": True},
    ]
    utts = reconstruct_speaker_utterances(units, source_language="fr")
    assert len(utts) == 3
    assert [u["source_text"] for u in utts] == ["Bonjour Marie.", "Salut Paul!", "Comment vas-tu?"]


def test_ctc_oov_chinese_failure_preserves_text_and_diarization_without_fabrication() -> None:
    # '钾' is known OOV in the checkpoint CTC vocabulary if checkpoint is present
    oov = find_ctc_oov_characters("<spk:0> 钙、钾等元素属于金属", frozenset("钙等元素属于金属"))
    assert oov == ["钾"]

    rec = recover_chunk_alignment(
        raw_text="<spk:0> 钙、钾等元素属于金属",
        ctc_payload={
            "words": [],
            "diarization": [{"speaker": 0, "start": 1.12, "end": 6.63}],
            "speaker_tag_to_diarization_speaker": {},
            "error": "alignment_failed",
        },
        source_offset_s=0.0,
        chunk_index=0,
    )
    assert rec["alignment_status"] == "diarization_bounded_only"
    assert len(rec["diarization"]) == 1
    assert all(u["start"] is None and u["end"] is None and not u["valid_timing"] for u in rec["units"])
    assert "".join(u["text"] for u in rec["units"]) == "钙、钾等元素属于金属"


def test_native_ctc_preserves_distinct_valid_observation_times() -> None:
    rec = recover_chunk_alignment(
        raw_text="<spk:0> Bonjour Merci",
        ctc_payload={"words": [
            {"word": "Bonjour", "start": 1.0, "end": 2.0, "speaker": "0"},
            {"word": "Merci", "start": 8.0, "end": 9.0, "speaker": "0"},
        ]},
        source_offset_s=0.0, chunk_index=0, model_path=None,
    )
    assert rec["alignment_status"] == "ctc_aligned"
    assert [u["text"] for u in rec["units"]] == ["Bonjour", "Merci"]
    assert [u["start"] for u in rec["units"]] == [1.0, 8.0]
    assert not rec["diagnostic"]["recovery_attempted"]


def test_partial_native_ctc_preserves_oov_and_usable_neighbors() -> None:
    rec = recover_chunk_alignment(
        raw_text="<spk:0> 你好。钾金属。",
        ctc_payload={"words": [
            {"word": "你", "start": 1.0, "end": 1.2, "speaker": "0"},
            {"word": "好。", "start": 1.2, "end": 1.4, "speaker": "0"},
        ]},
        source_offset_s=0.0, chunk_index=0, model_path=None,
    )
    assert rec["alignment_status"] == "partial"
    assert rec["diagnostic"]["untimed_units"] == 3
    assert [u["text"] for u in rec["units"]] == ["你", "好。", "钾", "金", "属。"]
    assert [u["valid_timing"] for u in rec["units"]] == [True, True, False, False, False]
    assert all(u["start"] is None for u in rec["units"][2:])


def test_cross_chunk_speaker_reconciliation_aba_and_collisions() -> None:
    chunks = plan_chunks(180.0, ChunkConfig(chunk_seconds=90.0, context_seconds=10.0, min_tail_seconds=15.0))
    # Chunk 0: local spk_0 is Alice (0..92s), local spk_1 is Bob (85..95s)
    # Chunk 1: local spk_0 is Bob (85..140s) -- NOTE numeric tag 0 swapped! -- local spk_1 is Alice (88..92s & 150..175s)
    u0, p0 = normalize_chunk_speakers(
        chunk_index=0,
        units=[
            {"text": "salut", "start": 86.0, "end": 86.5, "speaker": "1", "unit_type": "word", "valid_timing": True},
            {"text": "alice", "start": 89.0, "end": 89.5, "speaker": "0", "unit_type": "word", "valid_timing": True},
        ],
        diarization=[
            {"speaker": 0, "start": 10.0, "end": 92.0, "valid_timing": True},
            {"speaker": 1, "start": 85.0, "end": 95.0, "valid_timing": True},
        ],
        speaker_mapping={"0": "0", "1": "1"},
    )
    u1, p1 = normalize_chunk_speakers(
        chunk_index=1,
        units=[
            {"text": "salut", "start": 86.02, "end": 86.52, "speaker": "0", "unit_type": "word", "valid_timing": True},
            {"text": "alice", "start": 89.02, "end": 89.52, "speaker": "1", "unit_type": "word", "valid_timing": True},
        ],
        diarization=[
            {"speaker": 0, "start": 85.0, "end": 140.0, "valid_timing": True},
            {"speaker": 1, "start": 88.0, "end": 92.0, "valid_timing": True},
        ],
        speaker_mapping={"0": "0", "1": "1"},
    )
    res = reconcile_speakers(chunks=chunks, chunk_profiles=[p0, p1], embedder=None)
    l2g = res["local_to_global"]
    # chunk_0:spk_0 (Alice) MUST match chunk_1:spk_1 (Alice), NOT chunk_1:spk_0 (Bob)!
    assert l2g["chunk_000:spk_0"] == l2g["chunk_001:spk_1"]
    assert l2g["chunk_000:spk_1"] == l2g["chunk_001:spk_0"]
    assert l2g["chunk_000:spk_0"] != l2g["chunk_000:spk_1"]
    assert res["reconciliation"]["reconciled_global_speaker_count"] == 2
    assert len(res["reconciliation"]["concurrent_speaker_pairs"]) >= 1


def test_unknown_speaker_not_collapsed_into_speaker_zero() -> None:
    chunks = plan_chunks(40.0)
    u0, p0 = normalize_chunk_speakers(
        chunk_index=0,
        units=[{"text": "bonjour", "start": 1.0, "end": 1.5, "speaker": None, "unit_type": "word", "valid_timing": True}],
        diarization=[],
        speaker_mapping={},
    )
    res = reconcile_speakers(chunks=chunks, chunk_profiles=[p0], embedder=None)
    assert res["local_to_global"]["chunk_000:unknown"] == "SPK_UNK_000"
    assert res["reconciliation"]["unknown_speaker_count"] == 1
