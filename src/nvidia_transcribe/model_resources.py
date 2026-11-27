"""Host model-resource bindings, independent of processing-task semantics.

This module performs configuration resolution only. It does not instantiate models,
contact endpoints, download checkpoints, or control GPU services. Adapters own
protocol-specific behavior; a role declares the adapters it accepts.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit


class ResourceConfigError(ValueError):
    """Invalid, absent, or incompatible model-resource configuration."""


# Stage requirements, not model names. Specialized adapters remain distinct.
ROLE_ADAPTERS: dict[str, frozenset[str]] = {
    "asr": frozenset({"moss_events"}),
    "speech_activity": frozenset({"firered_aed"}),
    "source_refinement": frozenset({"openai_chat"}),
    "text_translation": frozenset({"openai_chat"}),
    "translation_review": frozenset({"openai_chat"}),
    "visual_inspection": frozenset({"openai_chat"}),
    "subtitle_condensation": frozenset({"openai_chat"}),
    "riva_translation": frozenset({"riva_worker"}),
}
RESOURCE_FIELDS = frozenset({"adapter", "model", "endpoint", "checkpoint_path", "worker_python", "device", "api_key_env"})
# A field that the selected adapter ignores must be rejected rather than silently
# suggesting it controls inference (particularly device and credential settings).
_ADAPTER_FIELDS = {
    "firered_aed": frozenset({"adapter", "model", "checkpoint_path", "worker_python", "device"}),
    "moss_events": frozenset({"adapter", "model", "checkpoint_path", "worker_python", "device"}),
    "openai_chat": frozenset({"adapter", "endpoint", "model", "api_key_env"}),
    "riva_worker": frozenset({"adapter", "checkpoint_path", "worker_python", "device"}),
}
_RESOURCE_DEFAULTS: dict[str, dict[str, str]] = {
    "moss_local": {
        "adapter": "moss_events",
        "model": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
        "checkpoint_path": "/workspace/models/MOSS-Transcribe-Diarize",
        "worker_python": "/workspace/nvidia-transcribe/.venv-moss/bin/python",
        "device": "cuda:0",
    },
    "text_local": {
        "adapter": "openai_chat",
        "endpoint": "http://127.0.0.1:8001/v1",
        "model": "chat",
    },
    "riva_local": {
        "adapter": "riva_worker",
        "checkpoint_path": "/workspace/models/Riva-Translate-4B-Instruct-v2",
        "worker_python": "/workspace/nvidia-transcribe/outputs/modular-eval/.venv/bin/python",
        "device": "cuda:0",
    },
}
_ROLE_DEFAULTS = {
    "asr": "moss_local",
    "source_refinement": "text_local",
    "text_translation": "text_local",
    "subtitle_condensation": "text_local",
    "translation_review": "text_local",
    "riva_translation": "riva_local",
}
_ID = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_ENV = re.compile(r"^[A-Z_][A-Z0-9_]*$")


@dataclass(frozen=True)
class ModelResource:
    name: str
    adapter: str
    model: str | None = None
    endpoint: str | None = None
    checkpoint_path: str | None = None
    worker_python: str | None = None
    device: str | None = None
    api_key_env: str | None = None

    def fingerprint(self) -> str:
        """Stable cache discriminator; never includes credential values."""
        fields = (self.adapter, self.model, self.endpoint, self.checkpoint_path,
                  self.worker_python, self.device, self.api_key_env)
        return hashlib.sha256(json.dumps(fields, separators=(",", ":")).encode()).hexdigest()

    def credential(self, environ: Mapping[str, str] | None = None) -> str | None:
        if not self.api_key_env:
            return None
        env = os.environ if environ is None else environ
        secret = env.get(self.api_key_env)
        if not secret:
            raise ResourceConfigError(
                f"Resource {self.name!r} requires environment variable {self.api_key_env}")
        return secret


def _validate_resource(name: str, spec: Mapping[str, Any]) -> ModelResource:
    if not isinstance(name, str) or not _ID.fullmatch(name):
        raise ResourceConfigError(f"Invalid resource name {name!r}")
    unexpected = set(spec) - RESOURCE_FIELDS
    if unexpected:
        raise ResourceConfigError(f"Resource {name!r} has unsupported keys: {sorted(unexpected)}")
    if not all(isinstance(v, str) and v.strip() for v in spec.values()):
        raise ResourceConfigError(f"Resource {name!r} fields must be nonempty strings")
    adapter = spec.get("adapter")
    if adapter not in _ADAPTER_FIELDS:
        raise ResourceConfigError(f"Resource {name!r} has unknown adapter {adapter!r}")
    unused = set(spec) - _ADAPTER_FIELDS[adapter]
    if unused:
        raise ResourceConfigError(
            f"Resource {name!r} adapter {adapter!r} cannot use fields: {sorted(unused)}")
    endpoint = spec.get("endpoint")
    if endpoint is not None:
        parsed = urlsplit(endpoint)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname or
                parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ResourceConfigError(f"Resource {name!r} endpoint must be an HTTP(S) base URL without credentials/query")
        try:
            _ = parsed.port
        except ValueError as exc:
            raise ResourceConfigError(f"Resource {name!r} has invalid endpoint port") from exc
    if adapter == "openai_chat" and not (endpoint and spec.get("model")):
        raise ResourceConfigError(f"Resource {name!r} requires endpoint and served model")
    if adapter == "riva_worker" and not (spec.get("checkpoint_path") and spec.get("worker_python")):
        raise ResourceConfigError(f"Resource {name!r} requires checkpoint_path and worker_python")
    if adapter == "moss_events" and not (spec.get("model") and spec.get("checkpoint_path")
            and spec.get("worker_python") and spec.get("device") == "cuda:0"):
        raise ResourceConfigError("MOSS requires model, local checkpoint_path, worker_python and explicit cuda:0")
    if adapter == "firered_aed" and not (spec.get("model") and spec.get("checkpoint_path")
            and spec.get("worker_python") and spec.get("device") == "cpu"):
        raise ResourceConfigError("FireRed AED requires model, local checkpoint_path, worker_python and explicit cpu")
    env_name = spec.get("api_key_env")
    if env_name and not _ENV.fullmatch(env_name):
        raise ResourceConfigError(f"Resource {name!r} api_key_env must name an environment variable")
    if adapter == "riva_worker":
        device = spec.get("device", "auto")
        if device not in ("auto", "cpu") and not re.fullmatch(r"cuda:\d+", device):
            raise ResourceConfigError("Riva worker device must be auto, cpu, or cuda:N")
    return ModelResource(name=name, **dict(spec))


@dataclass(frozen=True)
class ModelResources:
    resources: Mapping[str, ModelResource]
    roles: Mapping[str, str]
    visual_inherits_review: bool = True

    def resolve(self, role: str, *, resource_name: str | None = None,
                endpoint: str | None = None, model: str | None = None,
                require_local_paths: bool = False) -> ModelResource:
        if role not in ROLE_ADAPTERS:
            raise ResourceConfigError(f"Unknown task role {role!r}")
        name = resource_name if resource_name is not None else self.roles.get(role)
        if not isinstance(name, str) or not _ID.fullmatch(name):
            raise ResourceConfigError(f"Role {role!r} has invalid resource name {name!r}")
        if name not in self.resources:
            raise ResourceConfigError(f"Role {role!r} references unknown resource {name!r}")
        resource = self.resources[name]
        if resource.adapter not in ROLE_ADAPTERS[role]:
            raise ResourceConfigError(f"Role {role!r} cannot use {resource.adapter!r} resource {name!r}")
        if resource.adapter != "openai_chat" and (endpoint is not None or model is not None):
            raise ResourceConfigError("Select a configured local worker resource instead of endpoint/model overrides")
        updates = {}
        if endpoint is not None:
            updates["endpoint"] = endpoint
        if model is not None:
            updates["model"] = model
        if updates:
            fields = {
                k: getattr(resource, k) for k in RESOURCE_FIELDS if getattr(resource, k) is not None
            }
            resource = _validate_resource(resource.name, {**fields, **updates})
        if require_local_paths:
            for attr in ("checkpoint_path", "worker_python"):
                location = getattr(resource, attr)
                if location and not Path(location).exists():
                    raise ResourceConfigError(
                        f"Resource {name!r} {attr} is unavailable (configure an installed local path)")
        return resource

    def public_bindings(self) -> dict[str, dict[str, str]]:
        """Troubleshooting inventory with no tokens, endpoint hostnames, or local paths."""
        result = {}
        for role, name in self.roles.items():
            resource = self.resolve(role)
            result[role] = {"resource": name, "adapter": resource.adapter, "fingerprint": resource.fingerprint()[:12]}
        return result


def load_model_resources(
    path: str | Path | None = None, *, environ: Mapping[str, str] | None = None
) -> ModelResources:
    """Resolve defaults < JSON host file < environment; CLI overrides happen later.

    TEXT_* governs only the built-in text_local binding. Independently named
    role-specific resources in the JSON file remain independent.
    """
    env = os.environ if environ is None else environ
    if {"ASR_BASE_URL", "ASR_MODEL"} & env.keys():
        raise ResourceConfigError("Retired ASR endpoint overrides; configure a MOSS local worker resource")
    if {"SWIFTU_BASE_URL", "SWIFTU_MODEL"} & env.keys():
        raise ResourceConfigError("Retired SWIFTU_* environment settings; use TEXT_BASE_URL and TEXT_MODEL")
    requested = path if path is not None else env.get("NEMOVOX_MODEL_RESOURCES")
    resources = {name: dict(spec) for name, spec in _RESOURCE_DEFAULTS.items()}
    roles = dict(_ROLE_DEFAULTS)
    if requested:
        config_path = Path(requested).expanduser()
        try:
            doc = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ResourceConfigError(f"Cannot load model resource config {config_path.name!r}: {exc}") from exc
        if not isinstance(doc, dict) or set(doc) - {"version", "resources", "roles"}:
            raise ResourceConfigError("Resource config must contain only version, resources, roles")
        if doc.get("version") != 1:
            raise ResourceConfigError("Unsupported model resource config version; expected 1")
        overrides = doc.get("resources", {})
        role_overrides = doc.get("roles", {})
        if not isinstance(overrides, dict) or not isinstance(role_overrides, dict):
            raise ResourceConfigError("resources and roles must be objects")
        for name, spec in overrides.items():
            if not isinstance(spec, dict):
                raise ResourceConfigError(f"Resource {name!r} definition must be an object")
            prior = resources.get(name, {})
            merged = {**prior, **spec}
            if name == "moss_local" and "model" in spec and spec["model"] != prior.get("model"):
                if "checkpoint_path" not in spec or "worker_python" not in spec:
                    raise ResourceConfigError("A different local recognition model requires explicit checkpoint and worker paths")
            resources[name] = merged
        if set(role_overrides) - ROLE_ADAPTERS.keys():
            raise ResourceConfigError(f"Unknown roles: {sorted(set(role_overrides) - ROLE_ADAPTERS.keys())}")
        for role, name in role_overrides.items():
            if not isinstance(name, str) or not _ID.fullmatch(name):
                raise ResourceConfigError(f"Role {role!r} must reference a valid resource name")
        roles.update(role_overrides)

    # Environment overrides are scoped to *default resource identities* only.
    for resource_name, settings in {
        "text_local": {"endpoint": env.get("TEXT_BASE_URL"),
                       "model": env.get("TEXT_MODEL")},
    }.items():
        for field_name, value in settings.items():
            if value is not None:
                if not value.strip():
                    raise ResourceConfigError(f"Empty environment override for {resource_name}.{field_name}")
                chosen = value.strip()
                resources[resource_name][field_name] = chosen
    # Inherit the selected review resource unless vision is explicitly bound.
    visual_inherits_review = "visual_inspection" not in roles
    roles.setdefault("visual_inspection", roles["translation_review"])
    validated = {name: _validate_resource(name, spec) for name, spec in resources.items()}
    registry = ModelResources(resources=validated, roles=roles, visual_inherits_review=visual_inherits_review)
    for role in ROLE_ADAPTERS:
        if role == "speech_activity" and role not in roles:
            continue
        registry.resolve(role)
    return registry
