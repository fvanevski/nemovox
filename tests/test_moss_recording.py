"""Native event invariants, no live GPU required."""
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from nvidia_transcribe import moss_recording as moss
from nvidia_transcribe.model_resources import ModelResources, ResourceConfigError, _validate_resource


def generation(text, count=120):
    return {"text": text, "generated_tokens": count, "max_new_tokens": 4096,
            "prompt_len": 200, "max_input_tokens": 131072}


def build(text):
    return moss.build_native_recording({"path": "clip.wav"}, "zh", {"model": "local"}, generation(text), 30.0, "audio")


def test_contiguous_native_events_preserve_speech_repeats_overlap_and_scope():
    raw = "[0.75][S01]完整句子[4.67][2.00][S02]插话[3.50][5.15][S01]完整句子[12.21]"
    c = build(raw)
    moss.validate_moss_contract(c)
    assert [r["source_text"] for r in c["utterances"]] == ["完整句子", "插话", "完整句子"]
    assert [r["speaker_id"] for r in c["utterances"]] == ["S01", "S02", "S01"]
    assert [r["source_start"] for r in c["utterances"]] == [0.75, 2.0, 5.15]
    assert all(not r["source_words"] and not r["source_characters"] and not r["source_unit_ids"]
               for r in c["utterances"])
    assert c["native_dialogue"]["coverage_verified"] is False


def test_small_end_quantization_and_material_invalid_intervals():
    c = build("[1][S01] Bonjour [2][4][S02] erreur [4][28.12][S02] fin [30.04][29][S01] hors [31]")
    moss.validate_moss_contract(c)
    by_text = {r["source_text"]: r for r in c["utterances"]}
    assert by_text["fin"]["source_end"] == 30
    assert by_text["fin"]["timestamp_provenance"]["raw_end"] == "30.04"
    assert by_text["erreur"]["source_start"] is None
    assert by_text["hors"]["source_end"] is None
    assert len(c["native_dialogue"]["untimed_event_ids"]) == 2


@pytest.mark.parametrize("raw", ["junk[1][S01]ok[2]", "[1][S01]ok[2]junk",
    "[1][S01]unfinished", "[1][S01][2][S02]x[3]", "[1][S01] [2]"])
def test_generation_cannot_silently_drop_unparsed_material(raw):
    with pytest.raises(ValueError):
        build(raw)


def test_token_budget_exhaustion_rejected():
    with pytest.raises(ValueError, match="capacity"):
        moss.build_native_recording({}, "fr", {}, generation("[1][S01]ok[2]",4096), 30, "audio")
    assert moss._token_budget(1800) > 4096


def test_retained_native_contract_rejects_invalid_times_and_fake_lexical_observations():
    c = build("[1][S01]Bonjour[2]")
    c["utterances"][0]["source_end"] = 31
    with pytest.raises(ValueError, match="bounds"):
        moss.validate_moss_contract(c)
    c = build("[1][S01]Bonjour[2]")
    c["utterances"][0]["source_unit_ids"] = ["invented"]
    with pytest.raises(ValueError, match="event"):
        moss.validate_moss_contract(c)


def test_resource_admission():
    spec = {"adapter": "moss_events", "model": "MOSS", "checkpoint_path": "/local/model",
            "worker_python": "/local/python", "device": "cuda:0"}
    r = _validate_resource("moss_local", spec)
    registry = ModelResources({"moss_local": r}, {"asr": "moss_local"})
    assert registry.resolve("asr") == r
    with pytest.raises(ResourceConfigError):
        registry.resolve("asr", endpoint="http://localhost:8000/v1")
    with pytest.raises(ResourceConfigError):
        _validate_resource("moss_local", {**spec, "device": "cpu"})


def test_checkpoint_reuses_and_refuses_modified_generation(tmp_path, monkeypatch):
    media_path = tmp_path / "input.wav"
    media_path.write_bytes(b"input")
    media = SimpleNamespace(path=str(media_path), duration_s=30.0, to_dict=lambda: {"path": str(media_path), "duration_s": 30})
    monkeypatch.setattr(moss, "probe_media", lambda *a, **k: media)
    monkeypatch.setattr(moss, "_resource_identity", lambda r: {"model": "test"})
    def extract(*a, **kw):
        kw["out_wav"].write_bytes(b"wav")
        return 30.0
    monkeypatch.setattr(moss, "extract_wav_segment", extract)
    monkeypatch.setattr(moss, "validate_wav_file", lambda *a, **k: (30.0, "pcm"))
    calls = []
    @contextmanager
    def worker(**kwargs):
        calls.append("lease")
        yield
    def run(args, **kwargs):
        calls.append("inference")
        Path(args[args.index("--output")+1]).write_text(json.dumps(generation("[1][S01]Bonjour[2]")))
    monkeypatch.setattr(moss.subprocess, "run", run)
    options = dict(output_dir=tmp_path / "out", resource=SimpleNamespace(adapter="moss_events",
        name="moss", checkpoint_path=str(tmp_path), worker_python="python"), lifecycle=SimpleNamespace(worker=worker),
        source_language="fr")
    first = moss.process_moss_media(media_path, **options)
    second = moss.process_moss_media(media_path, **options)
    assert first["contract"] == second["contract"]
    assert second["resume"] == "reused"
    assert calls == ["lease", "inference"]
    Path(first["native_generation_path"]).write_text("{}")
    with pytest.raises(ValueError, match="identity mismatch"):
        moss.process_moss_media(media_path, **options)
    assert calls == ["lease", "inference"]
