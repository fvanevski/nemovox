"""Cached source-evidence replay must read verified raw chunks, never infer success."""

import hashlib
import json
from pathlib import Path

import pytest

from nvidia_transcribe.source_event_replay import replay_cached_source_evidence


def _cache(tmp_path: Path, *, finish="stop", digest_override=None):
    root = tmp_path / "asr"
    (root / "chunks").mkdir(parents=True)
    raw = {
        "choices": [{"message": {"content": "<spk:0> bonjour monde"},
                     "finish_reason": finish}],
        "ctc_timestamps": {
            "words": [
                {"word": "bonjour", "start": 1.0, "end": 1.35, "speaker": "0"},
                {"word": "monde", "start": 1.4, "end": 1.8, "speaker": "0"},
            ],
            "diarization": [],
        },
    }
    dump = json.dumps(raw, ensure_ascii=False, indent=2)
    (root / "chunks" / "chunk_000.raw.json").write_text(dump, encoding="utf-8")
    manifest = {
        "asr_run_status": "completed",
        "media": {"fingerprint_sha256": "immut-media-id"},
        "chunks": [{
            "chunk_index": 0,
            "status": "completed",
            "core_start_s": 0.0, "core_end_s": 10.0,
            "infer_start_s": 0.0, "infer_end_s": 10.0,
            "raw_response_sha256": digest_override or hashlib.sha256(dump.encode()).hexdigest(),
        }],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_verified_offline_replay_preserves_raw_source_and_job_immutability(tmp_path):
    root = _cache(tmp_path)
    before = {p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.rglob("*") if p.is_file()}
    ledger = replay_cached_source_evidence(root)
    after = {p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in root.rglob("*") if p.is_file()}
    assert before == after
    assert ledger["mode"] == "shadow_observation_only"
    assert ledger["replay_provenance"]["mode"] == "offline_native_ctc_only"
    assert ledger["diagnostics"]["source_unit_observations"] == 2
    assert [u["text"] for u in ledger["lexical_observations"]] == ["bonjour", "monde"]
    assert all(not u["word_alignment_accepted"] for u in ledger["lexical_observations"])


def test_raw_digest_mismatch_is_a_hard_failure(tmp_path):
    root = _cache(tmp_path, digest_override="0" * 64)
    with pytest.raises(ValueError, match="digest mismatch"):
        replay_cached_source_evidence(root)


def test_incomplete_generation_is_not_valid_even_if_manifest_claims_complete(tmp_path):
    root = _cache(tmp_path, finish="length")
    with pytest.raises(ValueError, match="incomplete"):
        replay_cached_source_evidence(root)


def test_archived_manifest_raw_response_path_cannot_redirect_reads(tmp_path):
    root = _cache(tmp_path)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["chunks"][0]["raw_response_path"] = str(tmp_path / "outside.raw.json")
    path.write_text(json.dumps(manifest))
    ledger = replay_cached_source_evidence(root)
    assert ledger["diagnostics"]["source_unit_observations"] == 2
