"""Optional CPU speech-activity assessment bound to an admitted native recording.

No model is downloaded, no acoustic challenge is automatically scheduled and
no source event is changed. The default pipeline never invokes this worker.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from nvidia_transcribe.media import compute_file_sha256, extract_wav_segment, probe_media
from nvidia_transcribe.moss_recording import validate_moss_contract
from nvidia_transcribe.source_assessment import ACTIVITY_SCHEMA, assess_native_dialogue

_WORKER = Path(__file__).with_name("_speech_activity_worker.py")


def _run_worker(resource, *args, identity=False):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"}
    try:
        result = subprocess.run([resource.worker_python, str(_WORKER), *args],
                                env=env, check=True, capture_output=True, text=True,
                                timeout=60 if identity else 1800)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("FireRed AED worker identity/inference failed") from exc
    return result.stdout


def _identity(resource):
    if resource.adapter != "firered_aed" or resource.device != "cpu":
        raise ValueError("Acoustic assessment requires explicitly configured CPU FireRed AED")
    root = Path(resource.checkpoint_path).resolve(strict=True)
    if not root.is_dir() or not Path(resource.worker_python).is_file():
        raise ValueError("Install and configure the admitted local FireRed AED checkpoint and worker")
    assets = {p.relative_to(root).as_posix(): compute_file_sha256(p)
              for p in sorted(root.rglob("*")) if p.is_file()
              and not any(part.startswith(".") for part in p.relative_to(root).parts)}
    if not assets:
        raise ValueError("Empty FireRed AED checkpoint")
    return {"resource_fingerprint": resource.fingerprint(), "model": resource.model,
            "assets": assets, "runtime": json.loads(_run_worker(resource, "--identity", identity=True)),
            "worker_sha256": compute_file_sha256(_WORKER),
            "adapter_sha256": compute_file_sha256(Path(__file__))}


def _normalize(raw, *, duration, audio_sha, identity):
    # The detector's frame clock is quantized. Preserve raw observations,
    # allow only one small physical end quantization correction; reject
    # every other invalid interval rather than interpreting it as no speech.
    observed_duration = raw.get("duration_s")
    if (raw.get("device") != "cpu" or type(observed_duration) not in (int, float)
            or not math.isfinite(observed_duration) or abs(observed_duration-duration) > 0.05):
        raise ValueError("Detector device or recording duration mismatch")
    tracks = raw.get("event_intervals")
    if not isinstance(tracks, dict) or set(tracks) != {"speech", "singing", "music"}:
        raise ValueError("Detector must report all three event streams")
    events = {}
    for kind, intervals in tracks.items():
        if not isinstance(intervals, list):
            raise ValueError("Invalid detector interval stream")
        events[kind] = []
        for pair in intervals:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("Invalid detector interval")
            s, e = pair
            if not (type(s) in (float,int) and type(e) in (float,int)
                    and math.isfinite(s) and math.isfinite(e)
                    and 0 <= s < min(e, duration) and e <= duration + 0.05):
                raise ValueError("Detector interval outside recording")
            events[kind].append([s, min(e, duration)])
    return {"schema": ACTIVITY_SCHEMA, "decoded_duration_s": duration,
            "audio_sha256": audio_sha, "resource_identity": identity,
            "event_intervals": events}


def assess_recording_activity(contract, *, media_path, resource, output_dir, resume=True):
    validate_moss_contract(contract, media_path=media_path)
    source = contract["source"]
    native = contract["native_dialogue"]
    duration = native["decoded_duration_s"]
    identity = _identity(resource)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "speech-activity.raw.json"
    key = {"media_sha256": source["content_sha256"], "audio_sha256": native["audio_sha256"],
           "stream": source["selected_audio_stream"], "decoded_duration_s": duration, "resource": identity}
    if resume and path.is_file():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior.get("key") == key:
            activity = _normalize(prior["raw"], duration=duration,
                                  audio_sha=native["audio_sha256"], identity=identity)
            if prior.get("activity") != activity:
                raise ValueError("Retained acoustic observation identity mismatch")
            return assess_native_dialogue(contract, activity), "reused_activity"
    path.unlink(missing_ok=True)
    with TemporaryDirectory(prefix="speech-activity-", dir=out) as td:
        wav = Path(td) / "recording.wav"
        output = Path(td) / "observations.json"
        media = probe_media(media_path, audio_stream_index=source["selected_audio_stream"])
        actual = extract_wav_segment(media.path, start_s=0, duration_s=media.duration_s,
                                     audio_stream_index=source["selected_audio_stream"], out_wav=wav)
        if actual != duration or compute_file_sha256(wav) != native["audio_sha256"]:
            raise ValueError("Decoded audio does not match native recognition input")
        _run_worker(resource, "--checkpoint", str(Path(resource.checkpoint_path).resolve()),
                    "--audio", str(wav), "--output", str(output))
        raw = json.loads(output.read_text(encoding="utf-8"))
        if (compute_file_sha256(wav) != native["audio_sha256"] or
                compute_file_sha256(Path(media_path)) != source["content_sha256"] or
                _identity(resource) != identity):
            raise ValueError("Audio or detector resource changed during assessment")
        activity = _normalize(raw, duration=duration, audio_sha=native["audio_sha256"], identity=identity)
        report = assess_native_dialogue(contract, activity)
        tmp = path.with_name("." + path.name + ".tmp")
        tmp.write_text(json.dumps({"key": key, "raw": raw, "activity": activity},
                                  sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, path)
    return report, "computed_activity"
