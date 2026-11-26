"""Comprehensive CPU regression tests verifying all eight Phase A correctness invariants."""

from __future__ import annotations

import json
import time
import wave
from pathlib import Path

import pytest

from nvidia_transcribe.alignment import annotate_units_with_source_spacing, recover_chunk_alignment
from nvidia_transcribe.chunking import ChunkConfig, plan_chunks, reconcile_chunk_boundaries
from nvidia_transcribe.client import NemotronVLLMClient
from nvidia_transcribe.recording import RecordingPipelineConfig, process_media
from nvidia_transcribe.speakers import _best_bipartite_assignment, normalize_chunk_speakers, reconcile_speakers
from nvidia_transcribe.utterances import join_multilingual_units, reconstruct_speaker_utterances


# --- A1: Unsafe deduplication involving unknown speakers ---

def test_a1_unknown_speaker_never_deduplicates_or_deletes_simultaneous_speech() -> None:
    chunks = plan_chunks(180.0, ChunkConfig(chunk_seconds=90.0, context_seconds=10.0, min_tail_seconds=15.0))
    # Known SPK_00 (core 0) and unknown SPK_UNK_001 (core 1) say the exact same word across the boundary
    # within dt <= 0.85s -> BOTH core observations preserved + ambiguity logged!
    c0_units = [
        {"text": "oui", "start": 89.70, "end": 90.10, "global_speaker_id": "SPK_00", "valid_timing": True, "chunk_index": 0},
    ]
    c1_units = [
        {"text": "oui", "start": 90.02, "end": 90.38, "global_speaker_id": "SPK_UNK_001", "valid_timing": True, "chunk_index": 1},
    ]
    merged, diag = reconcile_chunk_boundaries([c0_units, c1_units], chunks)
    assert diag["deduplicated_units"] == 0
    assert [u["text"] for u in merged] == ["oui", "oui"]
    assert {u["global_speaker_id"] for u in merged} == {"SPK_00", "SPK_UNK_001"}
    assert len(diag["ambiguities"]) == 1
    assert diag["ambiguities"][0]["reason"] == "unknown_speaker_cannot_prove_duplicate"
    assert diag["ambiguities"][0]["resolution"] == "preserved_both"


def test_a1_same_speaker_repeated_words_and_untimed_preservation() -> None:
    chunks = plan_chunks(180.0, ChunkConfig(chunk_seconds=90.0, context_seconds=10.0, min_tail_seconds=15.0))
    # Same speaker says "non" twice 0.95s apart (> tol) OR closely with only one right-chunk copy -> preserves genuine repetition;
    # untimed units are preserved without deletion.
    c0_units = [
        {"text": "non", "start": 89.0, "end": 89.2, "global_speaker_id": "SPK_00", "valid_timing": True, "chunk_index": 0},
        {"text": "non", "start": 89.6, "end": 89.9, "global_speaker_id": "SPK_00", "valid_timing": True, "chunk_index": 0},
        {"text": "钾", "start": None, "end": None, "global_speaker_id": "SPK_00", "valid_timing": False, "chunk_index": 0},
    ]
    c1_units = [
        {"text": "non", "start": 89.62, "end": 89.91, "global_speaker_id": "SPK_00", "valid_timing": True, "chunk_index": 1},
    ]
    merged, diag = reconcile_chunk_boundaries([c0_units, c1_units], chunks)
    assert diag["deduplicated_units"] == 1
    assert [u["text"] for u in merged] == ["non", "non", "钾"]


# --- A2: Native timing observations cannot rewrite the recognized spine ---

def test_native_timing_observations_never_substitute_proper_nouns_or_insert_text() -> None:
    rec = recover_chunk_alignment(
        raw_text="<spk:0> Cité de Koumlar est Nat'cal",
        ctc_payload={"words": [
            {"word": word, "start": float(i + 1), "end": i + 1.2, "speaker": "0"}
            for i, word in enumerate(["Cité", "de", "Kumla", "extra", "Nat'cal"])
        ]},
        source_offset_s=0.0, chunk_index=0, model_path=None,
    )
    assert [u["text"] for u in rec["units"]] == ["Cité", "de", "Koumlar", "est", "Nat'cal"]
    assert [u["valid_timing"] for u in rec["units"]] == [True, True, False, False, True]
    assert all(rec["units"][i]["start"] is None for i in (2, 3))
    assert rec["alignment_status"] == "partial"
    observations = rec["diagnostic"]["native_ctc_observations"]
    assert [o["raw_word"]["word"] for o in observations] == ["Cité", "de", "Kumla", "extra", "Nat'cal"]
    assert all("source_unit_index" not in observations[i] for i in (2, 3))


# --- A3 & A4: Timed/untimed ordering & accurate partial-timing status ---

def test_a3_a4_mixed_timed_untimed_preserves_order_and_marks_partial_status() -> None:
    units = [
        {"text": "甲", "start": 0.1, "end": 0.3, "speaker": "0", "unit_type": "character", "valid_timing": True, "chunk_index": 0, "alignment_status": "ctc_aligned"},
        {"text": "乙", "start": None, "end": None, "speaker": "0", "unit_type": "character", "valid_timing": False, "timing_error": "oov", "chunk_index": 0, "alignment_status": "diarization_bounded_only"},
        {"text": "丙", "start": 0.5, "end": 0.7, "speaker": "0", "unit_type": "character", "valid_timing": True, "chunk_index": 0, "alignment_status": "ctc_aligned"},
    ]
    utts = reconstruct_speaker_utterances(units, source_language="zh")
    assert len(utts) == 1
    u = utts[0]
    # A3: Strictly preserves 甲乙丙, NEVER 甲丙乙!
    assert u["source_text"] == "甲乙丙"
    # A4: MUST NOT say ctc_aligned or timing_status='full' when 乙 is untimed!
    assert u["alignment_status"] == "partial_alignment"
    assert u["timestamp_provenance"]["timing_status"] == "partial"
    assert u["timestamp_provenance"]["is_partial_bounding_interval"] is True
    assert u["timed_unit_count"] == 2
    assert u["untimed_unit_count"] == 1
    assert len(u["timing_gaps"]) == 1
    assert u["timing_gaps"][0]["text"] == "乙"
    assert u["timing_gaps"][0]["bounded_after_s"] == 0.3
    assert u["timing_gaps"][0]["bounded_before_s"] == 0.5


# --- A5: Ambiguous speaker matches NOT automatically merged ---

def test_a5_ambiguous_boundary_and_embedding_matches_stay_unmerged() -> None:
    chunks = plan_chunks(180.0, ChunkConfig(chunk_seconds=90.0, context_seconds=10.0, min_tail_seconds=15.0))
    # chunk 0: spk_0 speaks in boundary overlap [85..95]
    # chunk 1: spk_0 and spk_1 both have identical overlap score with chunk_0:spk_0 -> equal/near-tie!
    _, p0 = normalize_chunk_speakers(
        chunk_index=0,
        units=[{"text": "salut", "start": 86.0, "end": 87.0, "speaker": "0", "unit_type": "word", "valid_timing": True}],
        diarization=[{"speaker": 0, "start": 85.0, "end": 95.0, "valid_timing": True}],
        speaker_mapping={"0": "0"},
    )
    _, p1 = normalize_chunk_speakers(
        chunk_index=1,
        units=[
            {"text": "salut", "start": 86.0, "end": 87.0, "speaker": "0", "unit_type": "word", "valid_timing": True},
            {"text": "salut", "start": 86.0, "end": 87.0, "speaker": "1", "unit_type": "word", "valid_timing": True},
        ],
        diarization=[
            {"speaker": 0, "start": 85.0, "end": 95.0, "valid_timing": True},
            {"speaker": 1, "start": 85.0, "end": 95.0, "valid_timing": True},
        ],
        speaker_mapping={"0": "0", "1": "1"},
    )
    res = reconcile_speakers(chunks=chunks, chunk_profiles=[p0, p1], embedder=None)
    l2g = res["local_to_global"]
    assert len(res["reconciliation"]["ambiguous_matches"]) >= 1
    # Invariant: ambiguous edges MUST NOT auto-merge chunk_0:spk_0 with either candidate!
    assert l2g["chunk_000:spk_0"] != l2g["chunk_001:spk_0"]
    assert l2g["chunk_000:spk_0"] != l2g["chunk_001:spk_1"]
    assert len(res["reconciliation"]["evidence"]) == 0


# --- A6: Stale extracted audio invalidation & source replacement ---

def test_a6_fresh_run_and_replaced_source_invalidate_stale_cached_wav(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import math, struct

    def write_tone_wav(p: Path, freq_hz: float, duration_s: float = 2.0) -> None:
        sr = 16000
        n = int(sr * duration_s)
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            frames = bytearray()
            for i in range(n):
                val = int(12000 * math.sin(2.0 * math.pi * freq_hz * i / sr))
                frames.extend(struct.pack("<h", val))
            wf.writeframes(bytes(frames))

    src = tmp_path / "media.wav"
    out_dir = tmp_path / "out"
    write_tone_wav(src, freq_hz=440.0, duration_s=2.0)

    seen_wav_bytes: list[bytes] = []

    def fake_transcribe(self, audio_path, config=None):
        seen_wav_bytes.append(Path(audio_path).read_bytes())
        return {
            "contract": {"raw_text": "<spk:0> un"},
            "raw_response": {
                "choices": [{"message": {"content": "<spk:0> un"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 5},
                "ctc_timestamps": {
                    "words": [{"word": "un", "start": 0.2, "end": 0.5, "speaker": "0"}],
                    "diarization": [{"speaker": 0, "start": 0.1, "end": 0.6}],
                    "speaker_tag_to_diarization_speaker": {"0": 0},
                    "error": None,
                },
            },
            "latency_s": 0.01,
        }

    monkeypatch.setattr(NemotronVLLMClient, "transcribe", fake_transcribe)
    client = NemotronVLLMClient(model="m")

    # 1. Initial run extracts 440Hz WAV
    process_media(src, output_dir=out_dir, client=client, config=RecordingPipelineConfig(source_language="fr", speaker_embeddings="none", resume=True))
    first_chunk_bytes = (out_dir / "chunks/chunk_000.wav").read_bytes()
    assert len(seen_wav_bytes) == 1

    # 2. Replace media AT THE EXACT SAME PATH with a 880Hz WAV of identical length!
    write_tone_wav(src, freq_hz=880.0, duration_s=2.0)
    process_media(src, output_dir=out_dir, client=client, config=RecordingPipelineConfig(source_language="fr", speaker_embeddings="none", resume=True))
    second_chunk_bytes = (out_dir / "chunks/chunk_000.wav").read_bytes()
    assert second_chunk_bytes != first_chunk_bytes
    assert len(seen_wav_bytes) == 2

    # 3. Corrupt cached WAV on disk while keeping manifest -> resume=True detects hash mismatch and re-extracts
    (out_dir / "chunks/chunk_000.wav").write_bytes(b"corrupt_wav_payload_1234567890" * 5)
    process_media(src, output_dir=out_dir, client=client, config=RecordingPipelineConfig(source_language="fr", speaker_embeddings="none", resume=True))
    assert (out_dir / "chunks/chunk_000.wav").read_bytes() == second_chunk_bytes
    assert len(seen_wav_bytes) == 3

    # 4. resume=False forces fresh re-extraction & fresh inference
    process_media(src, output_dir=out_dir, client=client, config=RecordingPipelineConfig(source_language="fr", speaker_embeddings="none", resume=False))
    assert len(seen_wav_bytes) == 4


# --- A7: Latin text corrupted inside Chinese utterances ---

def test_a7_latin_spans_preserved_inside_chinese_using_source_provenance() -> None:
    raw = "<spk:0> Schotterb四零三路公交车 U S A 测试"
    ctc_units = [
        {"text": ch, "unit_type": "word", "speaker": "0", "valid_timing": True, "start": 0.1 * i, "end": 0.1 * (i + 1)}
        for i, ch in enumerate(["S", "c", "h", "o", "t", "t", "e", "r", "b"])
    ] + [
        {"text": ch, "unit_type": "character", "speaker": "0", "valid_timing": True, "start": 1.0 + 0.1 * i, "end": 1.1 + 0.1 * i}
        for i, ch in enumerate("四零三路公交车")
    ] + [
        {"text": "U", "unit_type": "word", "speaker": "0", "valid_timing": True, "start": 2.0, "end": 2.1},
        {"text": "S", "unit_type": "word", "speaker": "0", "valid_timing": True, "start": 2.1, "end": 2.2},
        {"text": "A", "unit_type": "word", "speaker": "0", "valid_timing": True, "start": 2.2, "end": 2.3},
        {"text": "测", "unit_type": "character", "speaker": "0", "valid_timing": True, "start": 2.4, "end": 2.5},
        {"text": "试", "unit_type": "character", "speaker": "0", "valid_timing": True, "start": 2.5, "end": 2.6},
    ]
    annotated = annotate_units_with_source_spacing(ctc_units, raw)
    joined = join_multilingual_units(annotated)
    # Contiguous Latin "Schotterb" preserved WITHOUT spaces; explicitly spaced acronym "U S A" NOT rewritten!
    assert joined == "Schotterb四零三路公交车 U S A 测试"


# --- A8: Polynomial bipartite speaker assignment (scales 7x7 & 8x8+ without factorial hang) ---

def test_a8_polynomial_bipartite_assignment_7x7_and_8x8_fast_and_exact() -> None:
    left_8 = [f"L{i}" for i in range(8)]
    right_8 = [f"R{i}" for i in range(8)]
    scores: dict[tuple[str, str], float] = {}
    for i, l in enumerate(left_8):
        for j, r in enumerate(right_8):
            scores[(l, r)] = 0.9 if i == j else 0.1
    t0 = time.perf_counter()
    matched = _best_bipartite_assignment(scores, left_8, right_8)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 50.0  # Factorial would take minutes; polynomial runs in <1ms
    assert sorted((l, r) for l, r, _ in matched) == [(f"L{i}", f"R{i}") for i in range(8)]

    # Forbidden pair respected without suboptimal greedy trap:
    # L0 wants R0 (0.9) and R1 (0.7); L1 wants R0 (0.85) and R1 (0.1).
    # Optimal must assign (L0->R1, L1->R0) total=1.55 instead of greedy (L0->R0, L1->R1) total=1.0!
    m2 = _best_bipartite_assignment(
        {("L0", "R0"): 0.9, ("L0", "R1"): 0.7, ("L1", "R0"): 0.85, ("L1", "R1"): 0.1},
        ["L0", "L1"],
        ["R0", "R1"],
    )
    assert dict((l, r) for l, r, _ in m2) == {"L0": "R1", "L1": "R0"}
