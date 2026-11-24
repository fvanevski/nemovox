"""Offline rerender must never enter the inference workflow."""
import json
from pathlib import Path

import pytest

from nvidia_transcribe import cli, subtitle_pipeline as pipeline


@pytest.fixture
def prohibit_inference(monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("inference stage")
        pytest.fail("Offline rerender entered an inference stage")

    for name in ("managed_asr_client", "managed_text_client", "process_media",
                 "review_current_source", "refine_utterances_with_text_model",
                 "translate_utterances", "review_translated_utterances",
                 "condense_high_cps_cues_with_text_model"):
        monkeypatch.setattr(pipeline, name, forbidden)
    return calls


@pytest.mark.parametrize("input_kind", ["media", "recording", "none"])
def test_missing_translated_contract_fails_shared_pipeline_before_inference(
        tmp_path, input_kind, prohibit_inference):
    media = tmp_path / "synthetic.wav"
    media.touch()
    recording = tmp_path / "recording.contract.json"
    recording.write_text(json.dumps({"utterances": []}))
    out = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="Offline re-render requires"):
        pipeline.generate_subtitles(
            media if input_kind == "media" else None,
            recording_contract_path=recording if input_kind == "recording" else None,
            output_dir=out, config=pipeline.SubtitlePipelineConfig(re_render_only=True),
            lifecycle=object(), lifecycle_resources={"asr": object()})
    assert not out.exists()
    assert not prohibit_inference


@pytest.mark.parametrize("input_kind", ["media", "recording", "none"])
def test_cli_missing_translated_contract_rejects_before_model_acquisition(
        tmp_path, monkeypatch, capsys, input_kind, prohibit_inference):
    # Use the public CLI with default, read-only resource bindings.
    monkeypatch.delenv("NEMOVOX_MODEL_RESOURCES", raising=False)
    out = tmp_path / "missing"
    args = ["generate-subtitles", "--output-dir", str(out),
            "--source-language", "fr", "--re-render"]
    if input_kind == "media":
        media = tmp_path / "synthetic.wav"
        media.touch()
        args.insert(1, str(media))
    elif input_kind == "recording":
        recording = tmp_path / "recording.contract.json"
        recording.write_text(json.dumps({"source_language": "fr", "utterances": []}))
        args.extend(["--recording-contract", str(recording)])
    assert cli.main(args) == 1
    assert "Offline re-render requires" in capsys.readouterr().err
    assert not out.exists()
    assert not prohibit_inference


def test_existing_contract_rerenders_both_formats_without_inference_or_mutation(
        tmp_path, prohibit_inference):
    translated = {
        "schema_version": "nemotron-recording-contract/v1",
        "source": {"path": "/synthetic/example.wav", "duration_s": 10.0},
        "source_language": "fr", "target_language": "en",
        "utterances": [{"id": "u1", "source_text": "On y va !",
                        "translated_text": "Let's go!", "speaker_id": "SPK_A",
                        "source_start": 1.0, "source_end": 3.0}],
    }
    contract = tmp_path / "translated.contract.json"
    original_bytes = json.dumps(translated).encode()
    contract.write_bytes(original_bytes)
    result = pipeline.generate_subtitles(
        output_dir=tmp_path,
        config=pipeline.SubtitlePipelineConfig(re_render_only=True, formats=("srt", "ass")),
        lifecycle=object(), lifecycle_resources={"asr": object()})
    for fmt in ("srt", "ass"):
        assert "Let's go!" in Path(result["files"][fmt]).read_text(encoding="utf-8")
    assert contract.read_bytes() == original_bytes
    assert not prohibit_inference


@pytest.mark.parametrize("bad_input", ["directory", "invalid_json"])
def test_rerender_unusable_contract_does_not_fall_back_to_inference(
        tmp_path, bad_input, prohibit_inference):
    contract = tmp_path / "translated.contract.json"
    if bad_input == "directory":
        contract.mkdir()
        expected = FileNotFoundError
    else:
        contract.write_text("not JSON")
        expected = json.JSONDecodeError
    with pytest.raises(expected):
        pipeline.generate_subtitles(
            output_dir=tmp_path, config=pipeline.SubtitlePipelineConfig(re_render_only=True),
            lifecycle=object(), lifecycle_resources={"asr": object()})
    assert not prohibit_inference
