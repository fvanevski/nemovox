"""Recording-scoped native event acquisition; no synthetic lexical/CTC ledger."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from nvidia_transcribe.contract import normalize_language_code, build_recording_contract
from nvidia_transcribe.media import compute_file_sha256, extract_wav_segment, probe_media, validate_wav_file

ADAPTER = "moss_events"
SCHEMA = "nemovox-native-dialogue/v1"
# Native compact grammar, not a line-oriented parser. Numeric square brackets
# are reserved for times. Non-numeric annotation text is retained verbatim.
_NUMBER = r"[0-9]+(?:\.[0-9]+)?"
_START = re.compile(r"\s*\[(" + _NUMBER + r")\]\s*\[(S[0-9]+)\]")
_END = re.compile(r"\[(" + _NUMBER + r")\]")
_WORKER = Path(__file__).with_name("_moss_worker.py")


def _encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)


def _digest(value):
    return hashlib.sha256(_encode(value).encode("utf-8")).hexdigest()


def _atomic(path, value):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(value, encoding="utf-8")
    os.replace(tmp, path)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def is_moss_contract(contract):
    return isinstance(contract, dict) and (
        (contract.get("model_provenance") or {}).get("adapter") == ADAPTER
        or (contract.get("native_dialogue") or {}).get("schema") == SCHEMA)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def parse_dialogue(raw: str, duration: float):
    """Require full consumption. Keep positive overlaps/repeats and invalid timing evidence."""
    if not isinstance(raw, str) or not _number(duration) or duration <= 0:
        raise ValueError("Invalid MOSS generation or audio duration")
    rows = []
    pos = 0
    while raw[pos:].strip():
        start_match = _START.match(raw, pos)
        if start_match is None:
            raise ValueError(f"Malformed or unconsumed MOSS generation at character {pos}")
        end_match = _END.search(raw, start_match.end())
        if end_match is None:
            raise ValueError(f"Incomplete MOSS event at character {pos}")
        txt = raw[start_match.end():end_match.start()].strip()
        if not txt or re.search(r"\[S[0-9]+\]", txt):
            raise ValueError(f"Empty or malformed MOSS event at character {pos}")
        s, e = float(start_match.group(1)), float(end_match.group(1))
        label = start_match.group(2)
        reason = None
        # Native timestamps are quantized. Normalize only a small overrun
        # at the known physical end; preserve original interval in evidence.
        if not math.isfinite(s) or not math.isfinite(e) or not 0 <= s < e:
            reason = "non_positive_or_invalid_interval"
        elif s >= duration or e > duration + 0.1:
            reason = "outside_decoded_audio"
        elif e > duration:
            reason = "end_clamped_to_decoded_audio"
        valid = reason in (None, "end_clamped_to_decoded_audio")
        idx = len(rows)
        rows.append({
            "id": f"event-{idx:05d}", "source_text": txt,
            "speaker_id": label, "raw_speaker_tag": label, "local_speaker_id": label,
            "source_start": s if valid else None,
            "source_end": min(e, duration) if valid else None,
            "source_words": [], "source_characters": [],
            "unit_count": 0, "timed_unit_count": 0, "untimed_unit_count": 0,
            "alignment_status": "native_event" if valid else "unavailable",
            "timestamp_provenance": {
                "backend": ADAPTER, "timing_status": "observed_segment" if valid else "unavailable",
                "raw_start": start_match.group(1), "raw_end": end_match.group(1),
                "source_offset_s": 0, "native_event_index": idx,
                "raw_character_span": [start_match.start(), end_match.end()],
                "speaker_scope": "recording_request", "error": reason,
            },
            "chunk_indices": [0],
        })
        pos = end_match.end()
    return rows


def _token_budget(duration):
    # Length-scaled capacity rather than the short-clip 4096 cap. Hitting
    # the limit is an explicit failed acquisition, not accepted completeness.
    return min(65536, max(4096, math.ceil(duration * 30)))


def build_native_recording(source, language, resource_identity, generation, duration, audio_sha):
    generated = generation.get("generated_tokens")
    limit = generation.get("max_new_tokens")
    prompt = generation.get("prompt_len")
    input_limit = generation.get("max_input_tokens")
    if (type(generated) is not int or type(limit) is not int or not 0 <= generated < limit
            or type(prompt) is not int or type(input_limit) is not int or not 0 < prompt < input_limit):
        raise ValueError("MOSS generation capacity reached or missing generation accounting")
    rows = parse_dialogue(generation["text"], duration)
    speakers = [{"speaker_id": label, "global_speaker_id": label, "local_speaker_ids": [label],
                 "identity_status": "anonymous_recording_request"}
                for label in dict.fromkeys(row["speaker_id"] for row in rows)]
    invalid = [r["id"] for r in rows if r["source_start"] is None]
    clipped = [r["id"] for r in rows if r["timestamp_provenance"]["error"] == "end_clamped_to_decoded_audio"]
    # The original acquisition order and raw spans remain in the evidence.
    # Render in temporal order without deleting overlapping or repeated speech.
    rows.sort(key=lambda r: (r["source_start"] is None,
                             r["source_start"] if r["source_start"] is not None else 0,
                             r["timestamp_provenance"]["native_event_index"]))
    warnings = []
    if invalid:
        warnings.append("Unlocalized MOSS events retained: " + ", ".join(invalid))
    if clipped:
        warnings.append("Native end quantization clamped: " + ", ".join(clipped))
    if not rows:
        warnings.append("No recognized dialogue; absence of speech is not independently confirmed")
    result = build_recording_contract(
        source=source, source_language=language, language_provenance="explicit",
        model_provenance={"adapter": ADAPTER, "resource_identity": resource_identity,
                          "request_mode": "speaker", "alignment_fallback": "off"},
        chunk_config={"geometry": "whole_recording", "decoded_duration_s": duration},
        chunk_records=[{"chunk_index": 0, "infer_start_s": 0, "infer_end_s": duration,
                        "audio_sha256": audio_sha, "raw_response_sha256": _digest(generation)}],
        speakers=speakers, utterances=rows, events=[], alignment_diagnostics=[],
        speaker_reconciliation={"status": "native_recording_scope", "identity_verified": False},
        boundary_reconciliation={"status": "not_applicable_single_request"},
        processing_warnings=warnings,
    )
    result["native_dialogue"] = {
        "schema": SCHEMA, "decoded_duration_s": duration, "audio_sha256": audio_sha,
        "generation_sha256": _digest(generation), "event_count": len(rows),
        "untimed_event_ids": invalid, "clamped_event_ids": clipped,
        "coverage_verified": False, "speaker_scope": "recording_request",
    }
    result["source_quality"] = {"status": "not_run", "reason": "native_event_candidate_no_lexical_recovery"}
    return result


def validate_moss_contract(contract, *, media_path=None, source_language=None):
    native = contract.get("native_dialogue") or {}
    if native.get("schema") != SCHEMA or (contract.get("model_provenance") or {}).get("adapter") != ADAPTER:
        raise ValueError("Incomplete native MOSS recording identity")
    duration = native.get("decoded_duration_s")
    source = contract.get("source") or {}
    rows = contract.get("utterances")
    language = normalize_language_code(contract.get("source_language"))
    if (not _number(duration) or duration <= 0 or not isinstance(rows, list)
            or len(rows) != native.get("event_count") or not language
            or source_language is not None and normalize_language_code(source_language) != language):
        raise ValueError("Invalid native MOSS event contract or source language mismatch")
    if media_path is not None and compute_file_sha256(media_path) != source.get("content_sha256"):
        raise ValueError("MOSS recording media identity mismatch")
    ids = set()
    untimed = set()
    for r in rows:
        uid = r.get("id")
        s, e = r.get("source_start"), r.get("source_end")
        if (not isinstance(uid, str) or uid in ids or not isinstance(r.get("source_text"), str)
                or not r["source_text"].strip() or r.get("source_language") != language
                or not re.fullmatch(r"S[0-9]+", str(r.get("speaker_id", "")))
                or r.get("source_words") or r.get("source_characters") or r.get("source_unit_ids")
                or (r.get("timestamp_provenance") or {}).get("backend") != ADAPTER):
            raise ValueError("Invalid or duplicate native MOSS event")
        ids.add(uid)
        if s is None and e is None:
            untimed.add(uid)
        elif not (_number(s) and _number(e) and 0 <= s < e <= duration):
            raise ValueError("Invalid native event bounds")
    if untimed != set(native.get("untimed_event_ids", [])):
        raise ValueError("Native untimed event inventory mismatch")


def _resource_identity(resource):
    root = Path(resource.checkpoint_path).resolve(strict=True)
    python = Path(resource.worker_python)
    if not root.is_dir() or not python.is_file() or not (root / "config.json").is_file():
        raise ValueError("MOSS needs an installed local checkpoint and worker interpreter")
    assets = {}
    # The admitted HF snapshot is flat. Hash all visible top-level model
    # files, including custom code and actual weights, not cache metadata.
    for path in sorted(root.iterdir()):
        if not path.name.startswith(".") and path.is_file():
            assets[path.name] = compute_file_sha256(path)
    if not any(name.endswith(".safetensors") for name in assets):
        raise ValueError("MOSS local safetensors are unavailable")
    try:
        run = subprocess.run([str(python), str(_WORKER), "--identity"], check=True,
                             capture_output=True, text=True, timeout=60)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("MOSS worker identity probe failed") from exc
    return {"resource_fingerprint": resource.fingerprint(), "model": resource.model,
            "assets": assets, "runtime": json.loads(run.stdout),
            "worker_sha256": compute_file_sha256(_WORKER),
            "adapter_sha256": compute_file_sha256(__file__)}


def process_moss_media(media_path, *, output_dir, resource, lifecycle,
                       source_language, audio_stream_index=0, resume=True):
    """One native recording request, managed worker, content-bound same-job resume."""
    start = time.perf_counter()
    language = normalize_language_code(source_language)
    if not language:
        raise ValueError("Select the actual source language for MOSS recording recognition")
    if resource.adapter != ADAPTER or lifecycle is None:
        raise ValueError("MOSS requires an explicitly admitted local worker")
    media = probe_media(media_path, audio_stream_index=audio_stream_index)
    source = {**media.to_dict(), "content_sha256": compute_file_sha256(media.path)}
    identity = _resource_identity(resource)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cp = out / "recording.contract.json"
    manifest = out / "manifest.json"
    raw_path = out / "native-dialogue.raw.json"
    transcript = out / "transcript.txt"
    key = {"schema": SCHEMA, "media_sha256": source["content_sha256"],
           "stream": audio_stream_index, "language": language, "resource": identity,
           "geometry": "whole_recording"}
    reused = False
    if resume and any(p.exists() for p in (cp, manifest, raw_path)):
        if not all(p.is_file() for p in (cp, manifest, raw_path)):
            raise ValueError("Incomplete native recording checkpoint; use a fresh output directory or --fresh")
        m = _read(manifest)
        if (m.get("key") != key or m.get("recording_sha256") != compute_file_sha256(cp)
                or m.get("generation_file_sha256") != compute_file_sha256(raw_path)):
            raise ValueError("Native recording checkpoint identity mismatch; use --fresh")
        generation = _read(raw_path)
        contract = _read(cp)
        validate_moss_contract(contract, source_language=language)
        native = contract["native_dialogue"]
        expected = build_native_recording(source, language, identity, generation,
                                          m["decoded_duration_s"], m["audio_sha256"])
        if contract != expected or native["generation_sha256"] != _digest(generation):
            raise ValueError("Native recording checkpoint does not reproduce the admitted events")
        reused = True
    else:
        # Invalidate the success marker before executing a fresh acquisition.
        manifest.unlink(missing_ok=True)
        cp.unlink(missing_ok=True)
        transcript.unlink(missing_ok=True)
        raw_path.unlink(missing_ok=True)
        wav = out / "recording.wav"
        tmp = out / ".native-dialogue.worker.json"
        tmp.unlink(missing_ok=True)
        duration = extract_wav_segment(media.path, start_s=0, duration_s=media.duration_s,
                                       out_wav=wav, audio_stream_index=audio_stream_index)
        actual_duration, _ = validate_wav_file(wav)
        if duration != actual_duration:
            raise ValueError("Extracted audio duration mismatch")
        if duration > 5400:
            raise ValueError("MOSS whole-recording candidate supports at most 90 minutes")
        audio_sha = compute_file_sha256(wav)
        try:
            with lifecycle.worker(resource_name=resource.name, min_free_vram_mb=8500):
                try:
                    subprocess.run([resource.worker_python, str(_WORKER),
                        "--checkpoint", str(Path(resource.checkpoint_path).resolve()),
                        "--audio", str(wav), "--output", str(tmp),
                        "--max-new-tokens", str(_token_budget(duration))], check=True, timeout=3600)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    raise RuntimeError("MOSS worker failed or timed out; no fallback selected") from exc
            if compute_file_sha256(wav) != audio_sha or compute_file_sha256(media.path) != source["content_sha256"]:
                raise ValueError("Recording input changed during inference")
            generation = _read(tmp)
            _atomic(raw_path, _encode(generation))  # preserve even a rejected generation
            if generation.get("device") != "cuda:0" or generation.get("dtype") != "torch.bfloat16":
                raise ValueError("MOSS worker device/dtype receipt mismatch")
            if _resource_identity(resource) != identity:
                raise ValueError("MOSS runtime or checkpoint changed during inference")
            if generation.get("max_new_tokens") != _token_budget(duration):
                raise ValueError("MOSS generation budget mismatch")
            contract = build_native_recording(source, language, identity, generation, duration, audio_sha)
            validate_moss_contract(contract, source_language=language)
            _atomic(cp, _encode(contract))
            _atomic(manifest, _encode({"key": key, "recording_sha256": compute_file_sha256(cp),
                "generation_file_sha256": compute_file_sha256(raw_path),
                "decoded_duration_s": duration, "audio_sha256": audio_sha}))
        finally:
            tmp.unlink(missing_ok=True)
            wav.unlink(missing_ok=True)
    # Use the existing transcript presentation contract without CTC reconstruction.
    from nvidia_transcribe.recording import _format_transcript_txt
    _atomic(transcript, _format_transcript_txt(contract))
    return {"contract": contract, "contract_path": str(cp), "transcript_path": str(transcript),
            "manifest_path": str(manifest), "native_generation_path": str(raw_path),
            "resume": "reused" if reused else "computed",
            "total_wall_time_s": round(time.perf_counter() - start, 4)}
