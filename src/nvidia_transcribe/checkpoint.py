"""Local checkpoint inspection and validation for Nemotron 3.5 Transcribe NVFP4."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MODEL_PATH = Path("/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4")

EXPECTED_ARCHITECTURES = ("Nemotron3_5TranscribeForConditionalGeneration",)
EXPECTED_MODEL_TYPE = "nemo_speechlm"

REQUIRED_CONFIG_FILES = (
    "config.json",
    "hf_quant_config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
)


class CheckpointValidationError(ValueError):
    """Raised when a local Nemotron checkpoint fails structural or quantization validation."""


@dataclass(frozen=True)
class CheckpointSummary:
    """Validated metadata summary for a local Nemotron Transcribe checkpoint."""

    model_path: str
    model_type: str
    architectures: tuple[str, ...]
    quant_method: str | None
    quant_algo: str | None
    nvfp4_targets: int
    fp8_targets: int
    has_ctc_timestamps: bool
    ctc_shard: str | None
    weight_shards: tuple[str, ...]
    total_bytes: int
    total_parameters: int | None
    revision: str
    config_hashes: dict[str, str] = field(default_factory=dict)
    shard_sizes: dict[str, int] = field(default_factory=dict)
    missing_files: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "model_type": self.model_type,
            "architectures": list(self.architectures),
            "quant_method": self.quant_method,
            "quant_algo": self.quant_algo,
            "nvfp4_targets": self.nvfp4_targets,
            "fp8_targets": self.fp8_targets,
            "has_ctc_timestamps": self.has_ctc_timestamps,
            "ctc_shard": self.ctc_shard,
            "weight_shards": list(self.weight_shards),
            "total_bytes": self.total_bytes,
            "total_parameters": self.total_parameters,
            "revision": self.revision,
            "config_hashes": dict(self.config_hashes),
            "shard_sizes": dict(self.shard_sizes),
            "missing_files": list(self.missing_files),
        }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _detect_hf_revision(model_dir: Path) -> str:
    """Read local HF download metadata if present; return 'unknown' if absent or ambiguous."""
    dl_dir = model_dir / ".cache" / "huggingface" / "download"
    meta_file = dl_dir / "config.json.metadata"
    if meta_file.is_file():
        lines = [line.strip() for line in meta_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        # HF download metadata stores commit hash on line 0 when downloaded from Hub,
        # or blob/etag hash; only accept a 40-char hex git SHA if line 0 and line 1 differ
        # or look like commit+etag, otherwise report unknown if uncertain.
        if len(lines) >= 2 and len(lines[0]) == 40 and all(c in "0123456789abcdef" for c in lines[0].lower()):
            return lines[0]
    return "unknown"


def inspect_checkpoint(model_path: str | Path = DEFAULT_MODEL_PATH) -> CheckpointSummary:
    """Inspect a local Nemotron checkpoint directory without loading weights into memory."""
    path = Path(model_path)
    if not path.is_dir():
        raise CheckpointValidationError(f"Checkpoint directory does not exist: {path}")

    missing = [name for name in REQUIRED_CONFIG_FILES if not (path / name).is_file()]
    if missing:
        raise CheckpointValidationError(f"Checkpoint {path} is missing required files: {missing}")

    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    hf_quant = json.loads((path / "hf_quant_config.json").read_text(encoding="utf-8"))
    index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))

    model_type = str(config.get("model_type") or "")
    architectures = tuple(config.get("architectures") or ())
    quant_cfg = config.get("quantization_config") or {}
    quant_method = quant_cfg.get("quant_method")
    quant_algo = (hf_quant.get("quantization") or {}).get("quant_algo")

    quantized_layers = (hf_quant.get("quantization") or {}).get("quantized_layers") or {}
    nvfp4_targets = sum(1 for v in quantized_layers.values() if isinstance(v, dict) and v.get("quant_algo") == "NVFP4")
    fp8_targets = sum(1 for v in quantized_layers.values() if isinstance(v, dict) and v.get("quant_algo") == "FP8")

    weight_map = index.get("weight_map") or {}
    shards = tuple(sorted(set(weight_map.values())))
    missing_shards = [shard for shard in shards if not (path / shard).is_file()]

    ctc_shards = sorted(
        {shard for tensor_name, shard in weight_map.items() if tensor_name.startswith("ctc_timestamp.")}
    )
    ctc_shard = ctc_shards[0] if len(ctc_shards) == 1 else None
    has_ctc_timestamps = bool(config.get("ctc_timestamps")) and ctc_shard is not None

    config_hashes = {name: _sha256_file(path / name) for name in REQUIRED_CONFIG_FILES}
    llm_cfg = path / "llm_backbone" / "config.json"
    if llm_cfg.is_file():
        config_hashes["llm_backbone/config.json"] = _sha256_file(llm_cfg)

    shard_sizes = {
        shard: (path / shard).stat().st_size for shard in shards if (path / shard).is_file()
    }
    total_bytes = sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and ".cache" not in p.parts)
    total_params = (index.get("metadata") or {}).get("total_parameters")

    return CheckpointSummary(
        model_path=str(path),
        model_type=model_type,
        architectures=architectures,
        quant_method=quant_method,
        quant_algo=quant_algo,
        nvfp4_targets=nvfp4_targets,
        fp8_targets=fp8_targets,
        has_ctc_timestamps=has_ctc_timestamps,
        ctc_shard=ctc_shard,
        weight_shards=shards,
        total_bytes=total_bytes,
        total_parameters=int(total_params) if total_params is not None else None,
        revision=_detect_hf_revision(path),
        config_hashes=config_hashes,
        shard_sizes=shard_sizes,
        missing_files=tuple(missing_shards),
    )


def validate_checkpoint(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    *,
    require_nvfp4: bool = True,
    require_ctc: bool = True,
) -> CheckpointSummary:
    """Validate that a local checkpoint matches the Nemotron 3.5 Transcribe NVFP4 contract."""
    summary = inspect_checkpoint(model_path)
    if summary.missing_files:
        raise CheckpointValidationError(f"Missing weight shard files: {list(summary.missing_files)}")
    if summary.model_type != EXPECTED_MODEL_TYPE:
        raise CheckpointValidationError(
            f"Unexpected model_type {summary.model_type!r}; expected {EXPECTED_MODEL_TYPE!r}"
        )
    if summary.architectures != EXPECTED_ARCHITECTURES:
        raise CheckpointValidationError(
            f"Unexpected architectures {summary.architectures!r}; expected {EXPECTED_ARCHITECTURES!r}"
        )
    if require_nvfp4:
        if summary.quant_method != "modelopt" or summary.quant_algo != "MIXED_PRECISION":
            raise CheckpointValidationError(
                f"Expected modelopt MIXED_PRECISION quantization, got method={summary.quant_method!r}, "
                f"algo={summary.quant_algo!r}"
            )
        if summary.nvfp4_targets == 0 or summary.fp8_targets == 0:
            raise CheckpointValidationError(
                f"Expected both NVFP4 and FP8 quantized targets, got nvfp4={summary.nvfp4_targets}, "
                f"fp8={summary.fp8_targets}"
            )
    if require_ctc and not summary.has_ctc_timestamps:
        raise CheckpointValidationError("Checkpoint lacks valid ctc_timestamps config or single CTC shard")
    return summary
