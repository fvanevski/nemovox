"""Native source assessment never establishes unobserved speech."""
import copy
import json

import pytest

from nvidia_transcribe.moss_recording import build_native_recording
from nvidia_transcribe.source_assessment import ACTIVITY_SCHEMA, assess_native_dialogue
from nvidia_transcribe.speech_activity import _normalize


def contract():
    generation = {"text": "[5][S01]Bonjour[10][8][S02]Salut[12][20][S01]Oui[22]",
        "generated_tokens": 120, "max_new_tokens": 4096, "prompt_len": 200,
        "max_input_tokens": 131072, "device": "cuda:0", "dtype": "torch.bfloat16"}
    return build_native_recording({"path": "clip.wav"}, "fr", {"model": "local"},
                                  generation, 30.0, "audio")


def test_geometry_includes_both_boundaries_and_unions_overlaps():
    c = contract()
    original = copy.deepcopy(c)
    r = assess_native_dialogue(c)
    assert r["observed_event_intervals"] == [[5,12],[20,22]]
    assert r["uncovered_intervals"] == [[0,5],[12,20],[22,30]]
    assert r["observed_event_union_s"] == 9
    assert r["uncovered_duration_s"] == 21
    assert r["speaker_event_counts"] == {"S01":2,"S02":1}
    assert r["singleton_speaker_labels"] == ["S02"]
    assert r["speech_recall_verified"] is False
    assert r["investigation_nominations"] == []
    assert c == original


def test_speech_nominations_are_only_intersections_and_never_corrections():
    c = contract()
    original = copy.deepcopy(c)
    activity = {"schema": ACTIVITY_SCHEMA, "audio_sha256": "audio",
        "decoded_duration_s":30, "resource_identity":{"model":"detector"},
        "event_intervals":{"speech":[[4,7],[14,16]],"singing":[[0,3]],"music":[[14,15],[22,30]]}}
    r = assess_native_dialogue(c, activity)
    assert [(n["start_s"],n["end_s"]) for n in r["investigation_nominations"]] == [(4,5),(14,16)]
    assert r["investigation_nominations"][1]["music_overlap_s"] == 1
    assert all(n["status"] == "unresolved" and not n["source_truth_accepted"] for n in r["investigation_nominations"])
    assert r["applied_corrections"] == 0
    assert c == original
    activity["audio_sha256"] = "different"
    with pytest.raises(ValueError,match="bind"):
        assess_native_dialogue(c,activity)


@pytest.mark.parametrize("bad", [[True,2], [2,1], [0,31], [0,float("inf")]])
def test_activity_invalid_interval_not_silently_discarded(bad):
    raw = {"device":"cpu","duration_s":30,"event_intervals":{"speech":[bad],"singing":[],"music":[]}}
    with pytest.raises(ValueError):
        _normalize(raw,duration=30,audio_sha="audio",identity={})


def test_detector_end_quantization_preserves_raw():
    raw = {"device":"cpu","duration_s":30,"event_intervals":{"speech":[[29,30.04]],"singing":[],"music":[]}}
    a = _normalize(raw,duration=30,audio_sha="audio",identity={})
    assert a["event_intervals"]["speech"] == [[29,30]]
    assert raw["event_intervals"]["speech"] == [[29,30.04]]


def test_cli_geometry_requires_no_worker_or_registry(tmp_path, monkeypatch):
    from nvidia_transcribe import cli
    import nvidia_transcribe.speech_activity as activity
    monkeypatch.setattr(activity,"assess_recording_activity",lambda *a,**kw: pytest.fail("implicit detector"))
    monkeypatch.setattr(cli,"load_model_resources",lambda *a,**kw: pytest.fail("implicit registry"))
    c = tmp_path/"recording.contract.json"
    c.write_text(json.dumps(contract()))
    out = tmp_path/"out"
    assert cli.main(["assess-source","--recording-contract",str(c),"--output-dir",str(out)]) == 0
    assert json.loads((out/"source-quality.json").read_text())["scope"] == "geometry_only"


def test_activity_resume_requires_bound_audio_and_unchanged_receipt(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from nvidia_transcribe import speech_activity as sa
    from nvidia_transcribe.media import compute_file_sha256
    c = contract()
    media = tmp_path / "episode.wav"
    media.write_bytes(b"original-media")
    c["source"].update(content_sha256=compute_file_sha256(media), selected_audio_stream=0)
    pcm = tmp_path / "decoded.wav"
    pcm.write_bytes(b"decoded-audio")
    c["native_dialogue"]["audio_sha256"] = compute_file_sha256(pcm)
    identity = {"model": "admitted"}
    monkeypatch.setattr(sa, "_identity", lambda r: identity)
    monkeypatch.setattr(sa, "probe_media", lambda *a, **k: SimpleNamespace(path=str(media), duration_s=30))
    def decode(*a, **kw):
        kw["out_wav"].write_bytes(pcm.read_bytes())
        return 30
    monkeypatch.setattr(sa, "extract_wav_segment", decode)
    calls = []
    def run(resource, *args, **kwargs):
        from pathlib import Path
        calls.append("detect")
        Path(args[args.index("--output")+1]).write_text(json.dumps({
            "device":"cpu","duration_s":30,
            "event_intervals":{"speech":[[14,16]],"singing":[],"music":[]}}))
    monkeypatch.setattr(sa,"_run_worker",run)
    resource = SimpleNamespace(checkpoint_path=str(tmp_path))
    kwargs = dict(media_path=media, resource=resource, output_dir=tmp_path/"report")
    first, status = sa.assess_recording_activity(c, **kwargs)
    assert status == "computed_activity"
    second, status = sa.assess_recording_activity(c, **kwargs)
    assert status == "reused_activity" and first == second and calls == ["detect"]
    receipt = kwargs["output_dir"]/"speech-activity.raw.json"
    data = json.loads(receipt.read_text())
    data["activity"]["event_intervals"]["speech"] = []
    receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="identity mismatch"):
        sa.assess_recording_activity(c, **kwargs)
    assert calls == ["detect"]
    c["native_dialogue"]["audio_sha256"] = "not-the-audio"
    with pytest.raises(ValueError,match="Decoded audio"):
        sa.assess_recording_activity(c, **kwargs)
    assert calls == ["detect"]
