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


def test_language_replayed_from_same_media_recording_contract(tmp_path):
    root = _cache(tmp_path)
    (root / "recording.contract.json").write_text(json.dumps({
        "schema_version": "nemotron-recording-contract/v1",
        "source": {"fingerprint_sha256": "immut-media-id"},
        "source_language": "French",
    }), encoding="utf-8")
    ledger = replay_cached_source_evidence(root)
    assert ledger["source_language"] == "fr"
    assert ledger["replay_provenance"]["source_language_provenance"] == "recording_contract_verified"


def test_stale_recording_contract_fingerprint_rejected_even_with_language(tmp_path):
    root = _cache(tmp_path)
    (root / "recording.contract.json").write_text(json.dumps({
        "schema_version": "nemotron-recording-contract/v1",
        "source": {"fingerprint_sha256": "OTHER_MEDIA"},
        "source_language": "fr",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        replay_cached_source_evidence(root)


def test_manifest_contract_language_disagreement_rejected(tmp_path):
    root = _cache(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_language"] = "zh"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (root / "recording.contract.json").write_text(json.dumps({
        "schema_version": "nemotron-recording-contract/v1",
        "source": {"fingerprint_sha256": "immut-media-id"},
        "source_language": "fr",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="source language mismatch"):
        replay_cached_source_evidence(root)


def test_raw_digest_mismatch_is_a_hard_failure(tmp_path):
    root = _cache(tmp_path, digest_override="0" * 64)
    with pytest.raises(ValueError, match="digest mismatch"):
        replay_cached_source_evidence(root)


def test_incomplete_generation_is_not_valid_even_if_manifest_claims_complete(tmp_path):
    root = _cache(tmp_path, finish="length")
    with pytest.raises(ValueError, match="incomplete"):
        replay_cached_source_evidence(root)


def test_forensic_replay_exposes_incomplete_chunk_without_accepting_its_words(tmp_path):
    root = _cache(tmp_path, finish="length")
    ledger = replay_cached_source_evidence(root, forensic_incomplete=True)
    assert ledger["replay_provenance"]["mode"] == "offline_forensic_partial"
    assert ledger["replay_provenance"]["rejected_or_incomplete_chunks"][0]["chunk_index"] == 0
    assert ledger["lexical_observations"] == []
    assert ledger["diagnostics"]["source_coverage_certified"] is False


def test_recording_integration_emits_manifest_bound_sidecar_and_keeps_v1(tmp_path, monkeypatch):
    import wave

    from nvidia_transcribe.client import NemotronVLLMClient
    from nvidia_transcribe.recording import RecordingPipelineConfig, process_media

    wav = tmp_path / "source.wav"
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(bytes(64000))

    calls = []
    response = {
        "choices": [{"message": {"content": "<spk:0> Bonjour monde"}, "finish_reason": "stop"}],
        "ctc_timestamps": {
            "words": [{"word": "Bonjour", "start": 0.5, "end": 0.8, "speaker": "0"},
                      {"word": "monde", "start": 0.9, "end": 1.2, "speaker": "0"}],
            "diarization": [{"speaker": 0, "start": 0.5, "end": 1.2}],
            "speaker_tag_to_diarization_speaker": {"0": 0},
            "error": None,
        },
    }

    def fake(self, path, config=None):
        calls.append(str(path))
        return {"raw_response": response, "contract": {"raw_text": "Bonjour monde"}, "latency_s": 0.01}

    monkeypatch.setattr(NemotronVLLMClient, "transcribe", fake)
    out = tmp_path / "job"
    config = RecordingPipelineConfig(source_language="fr", speaker_embeddings="none")
    first = process_media(wav, output_dir=out, client=NemotronVLLMClient(model="fixture"), config=config)
    sidecar = out / "source-evidence.v1.json"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    direct = json.loads(sidecar.read_text(encoding="utf-8"))
    assert first["contract"]["schema_version"] == "nemotron-recording-contract/v1"
    assert first["source_evidence_path"] == str(sidecar)
    assert direct["diagnostics"]["source_unit_observations"] == 2
    assert manifest["source_evidence"]["sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
    assert manifest["source_evidence"]["mode"] == "shadow_observation_only"
    ids = [unit["id"] for unit in direct["lexical_observations"]]

    again = process_media(wav, output_dir=out, client=NemotronVLLMClient(model="fixture"), config=config)
    assert len(calls) == 1
    assert [unit["id"] for unit in json.loads(sidecar.read_text())["lexical_observations"]] == ids
    assert again["contract"]["schema_version"] == "nemotron-recording-contract/v1"


def test_archived_manifest_raw_response_path_cannot_redirect_reads(tmp_path):
    root = _cache(tmp_path)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["chunks"][0]["raw_response_path"] = str(tmp_path / "outside.raw.json")
    path.write_text(json.dumps(manifest))
    ledger = replay_cached_source_evidence(root)
    assert ledger["diagnostics"]["source_unit_observations"] == 2
