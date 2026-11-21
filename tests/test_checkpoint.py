"""CPU-only unit tests for local checkpoint configuration and identity validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nvidia_transcribe.checkpoint import (
    DEFAULT_MODEL_PATH,
    CheckpointValidationError,
    validate_checkpoint,
)


def _make_fake_checkpoint(tmp_path: Path, *, quant_method: str = "modelopt", quant_algo: str = "MIXED_PRECISION") -> Path:
    (tmp_path / "llm_backbone").mkdir()
    (tmp_path / "llm_backbone" / "config.json").write_text('{"model_type": "nemotron_h"}', encoding="utf-8")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "nemo_speechlm",
                "architectures": ["Nemotron3_5TranscribeForConditionalGeneration"],
                "quantization_config": {"quant_method": quant_method},
                "ctc_timestamps": {"decoder_config": {"num_classes": 100}},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps(
            {
                "quantization": {
                    "quant_algo": quant_algo,
                    "quantized_layers": {
                        "backbone.layers.0.mixer.in_proj": {"quant_algo": "FP8"},
                        "backbone.layers.1.mixer.experts": {"quant_algo": "NVFP4", "group_size": 16},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "generation_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "chat_template.jinja").write_text("", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_parameters": 1234},
                "weight_map": {
                    "language_model.weight": "model-00001-of-00001.safetensors",
                    "ctc_timestamp.decoder.weight": "model-ctc-bf16.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"stub")
    (tmp_path / "model-ctc-bf16.safetensors").write_bytes(b"stub-ctc")
    return tmp_path


def test_validate_checkpoint_accepts_valid_layout(tmp_path: Path) -> None:
    ckpt = _make_fake_checkpoint(tmp_path)
    summary = validate_checkpoint(ckpt)
    assert summary.model_type == "nemo_speechlm"
    assert summary.architectures == ("Nemotron3_5TranscribeForConditionalGeneration",)
    assert summary.quant_method == "modelopt"
    assert summary.quant_algo == "MIXED_PRECISION"
    assert summary.nvfp4_targets == 1
    assert summary.fp8_targets == 1
    assert summary.ctc_shard == "model-ctc-bf16.safetensors"
    assert summary.revision == "unknown"


def test_validate_checkpoint_rejects_bf16_non_nvfp4(tmp_path: Path) -> None:
    ckpt = _make_fake_checkpoint(tmp_path, quant_method="none", quant_algo="BF16")
    with pytest.raises(CheckpointValidationError, match="MIXED_PRECISION"):
        validate_checkpoint(ckpt, require_nvfp4=True)


def test_validate_checkpoint_rejects_missing_shard(tmp_path: Path) -> None:
    ckpt = _make_fake_checkpoint(tmp_path)
    (ckpt / "model-00001-of-00001.safetensors").unlink()
    with pytest.raises(CheckpointValidationError, match="Missing weight shard"):
        validate_checkpoint(ckpt)


def test_real_local_checkpoint_metadata_if_present() -> None:
    if not DEFAULT_MODEL_PATH.is_dir():
        pytest.skip("Local Nemotron checkpoint directory not present on host")
    summary = validate_checkpoint(DEFAULT_MODEL_PATH)
    assert summary.nvfp4_targets > 0
    assert summary.fp8_targets > 0
    assert summary.ctc_shard == "model-ctc-bf16.safetensors"
    assert summary.total_bytes > 10_000_000_000
