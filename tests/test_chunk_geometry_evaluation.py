"""CPU-only checks for the bounded acoustic evaluation's evidence contract."""
import hashlib
import json
import tarfile
from io import BytesIO

import pytest

from scripts.evaluate_chunk_geometry import (
    compare_reference, comparison_metrics, read_authority, validate_partition, validate_regions,
)
from nvidia_transcribe.source_quality import _lex


def event(uid, text, start, end):
    return {"observation_id": uid, "source_text": text, "source_start": start,
            "source_end": end, "local_voice": "SPK_0"}


def reference(uid, text, start, end):
    return {**event(uid, text, start, end), "supporting_observation_ids": [uid+"a", uid+"b"]}


def arm(rows):
    return {"utterances": [r for r in rows if r["source_start"] is not None],
            "all_rows": rows, "raw_turn_lexemes": [_lex(r["source_text"]) for r in rows]}


def test_match_uses_absolute_timing_and_retains_real_repetition():
    ref = [reference("ref1","Bonjour tout le monde",10.,12.),
           reference("ref2","Bonjour tout le monde",20.,22.)]
    selected = arm([event("a","Bonjour tout le monde",10.1,12.2),
                    event("b","Bonjour tout le monde",20.2,22.3)])
    decisions = compare_reference(ref, selected)
    assert [d["matched_ids"] for d in decisions] == [["a"],["b"]]
    assert comparison_metrics(decisions)["mean_abs_onset_delta_s"] == .15
    assert comparison_metrics(decisions)["mean_abs_end_delta_s"] == .25


def test_conflict_unlocalized_and_unmatched_are_not_proven_omissions():
    refs = [reference("r1","one sentence",10.,12.),
            reference("r2","another sentence",20.,22.),
            reference("r3","not exactly present",30.,32.)]
    decisions = compare_reference(refs, arm([
        event("a","one sentence",15.,17.),
        event("b","another sentence",None,None),
    ]))
    assert [d["status"] for d in decisions] == [
        "exact_text_different_timing_unresolved", "exact_text_unlocalized", "no_strict_event_match"]
    assert comparison_metrics(decisions)["mean_abs_onset_delta_s"] is None


def test_reused_reference_match_is_explicitly_ambiguous():
    refs = [reference("r1","same phrase",10.,12.),reference("r2","same phrase",10.2,12.2)]
    decisions = compare_reference(refs, arm([event("a","same phrase",10.1,12.1)]))
    assert all(d["status"] == "ambiguous_reused_match" and "matched_ids" not in d for d in decisions)


def test_partition_and_preregistered_regions():
    validate_regions(1379.901)
    with pytest.raises(ValueError):
        validate_regions(1200.)
    chunks = [{"chunk_index":0,"core_start_s":0.,"core_end_s":40.,"infer_start_s":0.,"infer_end_s":44.},
              {"chunk_index":1,"core_start_s":40.,"core_end_s":75.,"infer_start_s":36.,"infer_end_s":75.}]
    validate_partition(chunks,75.)
    chunks[1]["core_start_s"] = 41.
    with pytest.raises(ValueError,match="gap or overlap"):
        validate_partition(chunks,75.)


def test_diagnostic_authority_requires_real_raw_bytes_and_matching_media(tmp_path):
    path = tmp_path/"diagnostic.tar.gz"
    members = {}
    observations = []
    for i in range(16):
        raw = json.dumps({"choices":[{"index":i}],"ctc_timestamps":{},"model":"test"}).encode()
        members[f"chunk-{i}.json"] = raw
        observations.append({"chunk_index":i,"raw_response_sha256":hashlib.sha256(raw).hexdigest()})
    members["ledger.json"] = json.dumps({"asr_observations":observations,
        "media_fingerprint_sha256":"test-fingerprint","source_language":"fr"}).encode()
    members["recording.json"] = json.dumps({"source":{"fingerprint_sha256":"test-fingerprint",
        "selected_audio_stream":1,"duration_s":1379.901,"path":"/private/media.mkv"}}).encode()
    with tarfile.open(path,"w:gz") as archive:
        for name,data in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member,BytesIO(data))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert read_authority(path,digest)["selected_audio_stream"] == 1
    with pytest.raises(ValueError,match="digest mismatch"):
        read_authority(path,"0"*64)
