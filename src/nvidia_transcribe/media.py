"""Media inspection and bounded WAV segment extraction via FFprobe/FFmpeg."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class MediaInspectionError(RuntimeError):
    """Raised when input media cannot be probed, lacks an audio stream, or extraction fails."""


@dataclass(frozen=True)
class MediaInfo:
    path: str
    duration_s: float
    format_name: str
    size_bytes: int
    audio_stream_count: int
    selected_audio_stream: int
    audio_codec: str | None
    sample_rate_hz: int | None
    channels: int | None
    fingerprint_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "duration_s": self.duration_s,
            "format_name": self.format_name,
            "size_bytes": self.size_bytes,
            "audio_stream_count": self.audio_stream_count,
            "selected_audio_stream": self.selected_audio_stream,
            "audio_codec": self.audio_codec,
            "sample_rate_hz": self.sample_rate_hz,
            "channels": self.channels,
            "fingerprint_sha256": self.fingerprint_sha256,
        }


def _bounded_file_fingerprint(path: Path, sample_bytes: int = 1024 * 1024) -> str:
    """Compute a fast deterministic SHA-256 over file size + head/tail chunks for resume validation."""
    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(f"size:{size}:".encode("utf-8"))
    with path.open("rb") as f:
        h.update(f.read(sample_bytes))
        if size > sample_bytes:
            f.seek(max(0, size - sample_bytes))
            h.update(f.read(sample_bytes))
    return h.hexdigest()


def probe_media(media_path: str | Path, *, audio_stream_index: int = 0) -> MediaInfo:
    """Inspect an audio or video file with ffprobe and validate audio stream presence."""
    path = Path(media_path)
    if not path.is_file():
        raise MediaInspectionError(f"Input media file not found: {path}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        raw = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise MediaInspectionError("ffprobe binary is required on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise MediaInspectionError(f"ffprobe failed on {path}: {exc.stderr.strip()}") from exc

    data = json.loads(raw)
    fmt = data.get("format") or {}
    streams = data.get("streams") or []
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio_streams:
        raise MediaInspectionError(f"Media file contains no audio streams: {path}")
    if audio_stream_index < 0 or audio_stream_index >= len(audio_streams):
        raise MediaInspectionError(
            f"Requested audio_stream_index={audio_stream_index} out of range (found {len(audio_streams)} audio streams)"
        )

    sel = audio_streams[audio_stream_index]
    dur_raw = fmt.get("duration") or sel.get("duration")
    try:
        duration_s = float(dur_raw)
    except (TypeError, ValueError) as exc:
        raise MediaInspectionError(f"Could not determine media duration for {path}: {dur_raw!r}") from exc
    if duration_s <= 0.0:
        raise MediaInspectionError(f"Non-positive media duration ({duration_s}s) for {path}")

    sr_raw = sel.get("sample_rate")
    ch_raw = sel.get("channels")
    return MediaInfo(
        path=str(path.resolve()),
        duration_s=round(duration_s, 6),
        format_name=str(fmt.get("format_name") or path.suffix.lstrip(".")),
        size_bytes=path.stat().st_size,
        audio_stream_count=len(audio_streams),
        selected_audio_stream=audio_stream_index,
        audio_codec=sel.get("codec_name"),
        sample_rate_hz=int(sr_raw) if sr_raw is not None else None,
        channels=int(ch_raw) if ch_raw is not None else None,
        fingerprint_sha256=_bounded_file_fingerprint(path),
    )


def inspect_audio_streams(media_path: str | Path) -> list[dict[str, Any]]:
    """Return actual 0:a:N streams, not arbitrary container stream indices.

    This is a bounded metadata-only ffprobe read. The Studio API checks path
    authority separately before calling it, and never returns a host path.
    """
    path = Path(media_path)
    if not path.is_file():
        raise MediaInspectionError("Selected media file is unavailable")
    try:
        raw = subprocess.check_output(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_streams", str(path)],
            text=True, stderr=subprocess.PIPE, timeout=30,
        )
        document = json.loads(raw)
    except FileNotFoundError as exc:
        raise MediaInspectionError("ffprobe is required for audio stream inspection") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaInspectionError("Audio stream inspection timed out") from exc
    except subprocess.CalledProcessError as exc:
        raise MediaInspectionError("ffprobe could not inspect selected media") from exc
    except (json.JSONDecodeError, TypeError) as exc:
        raise MediaInspectionError("ffprobe returned invalid stream metadata") from exc
    result: list[dict[str, Any]] = []
    for stream in document.get("streams") or []:
        if not isinstance(stream, dict) or stream.get("codec_type") != "audio":
            continue
        tags = stream.get("tags") or {}
        tags = tags if isinstance(tags, dict) else {}
        def positive_int(value: Any) -> int | None:
            try:
                parsed = int(value)
            except (TypeError, ValueError, OverflowError):
                return None
            return parsed if parsed > 0 else None
        result.append({
            "index": len(result),
            "container_index": stream.get("index"),
            "codec": stream.get("codec_name"),
            "language": tags.get("language"),
            "title": tags.get("title"),
            "channels": positive_int(stream.get("channels")),
            "sample_rate_hz": positive_int(stream.get("sample_rate")),
        })
    if not result:
        raise MediaInspectionError("Selected media contains no audio tracks")
    return result


def has_visual_stream(media_path: str | Path) -> bool:
    """Probe the first actual video stream, excluding attached pictures.

    A successful empty stream list establishes absence; probe failure or
    malformed metadata never does. Match FFmpeg's 0:V:0 extraction selector.
    """
    try:
        raw = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "V:0",
             "-show_entries", "stream=index,codec_type", "-of", "json",
             str(media_path)],
            text=True, stderr=subprocess.PIPE, timeout=30,
        )
        document = json.loads(raw)
    except (OSError, subprocess.SubprocessError, ValueError, TypeError) as exc:
        raise MediaInspectionError("Could not inspect visual stream capability") from exc
    streams = document.get("streams") if isinstance(document, dict) else None
    if (not isinstance(streams, list) or len(streams) > 1
            or any(not isinstance(s, dict) or s.get("codec_type") != "video"
                   or type(s.get("index")) is not int or s["index"] < 0 for s in streams)):
        raise MediaInspectionError("Invalid visual stream metadata")
    return bool(streams)


def compute_file_sha256(path: Path) -> str:
    """Compute exact streaming SHA-256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_wav_file(wav_path: str | Path, *, expected_sample_rate_hz: int = 16000) -> tuple[float, str]:
    """Verify a WAV file is readable, non-empty, mono PCM, and return (duration_s, sha256)."""
    p = Path(wav_path)
    if not p.is_file() or p.stat().st_size <= 44:
        raise MediaInspectionError(f"WAV file missing or empty: {p}")
    try:
        with wave.open(str(p), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            channels = wf.getnchannels()
    except wave.Error as exc:
        raise MediaInspectionError(f"Corrupt WAV header in {p}: {exc}") from exc
    if frames <= 0 or rate <= 0:
        raise MediaInspectionError(f"WAV file has zero frames: {p}")
    if channels != 1 or rate != expected_sample_rate_hz:
        raise MediaInspectionError(
            f"WAV format mismatch in {p}: expected 1ch@{expected_sample_rate_hz}Hz, got {channels}ch@{rate}Hz"
        )
    return round(frames / float(rate), 6), compute_file_sha256(p)


def extract_wav_segment(
    media_path: str | Path,
    *,
    start_s: float,
    duration_s: float,
    out_wav: str | Path,
    sample_rate_hz: int = 16000,
    audio_stream_index: int = 0,
) -> float:
    """Extract a 16 kHz mono PCM16 WAV segment atomically via ffmpeg and return actual WAV duration."""
    if start_s < 0.0 or duration_s <= 0.0:
        raise ValueError(f"Invalid extraction window: start_s={start_s}, duration_s={duration_s}")
    out_path = Path(out_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp.wav")

    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-ss",
        f"{start_s:.6f}",
        "-t",
        f"{duration_s:.6f}",
        "-i",
        str(media_path),
        "-map",
        f"0:a:{audio_stream_index}",
        "-ac",
        "1",
        "-ar",
        str(sample_rate_hz),
        "-c:a",
        "pcm_s16le",
        str(tmp_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        tmp_path.unlink(missing_ok=True)
        raise MediaInspectionError("ffmpeg binary is required on PATH") from exc
    except subprocess.CalledProcessError as exc:
        tmp_path.unlink(missing_ok=True)
        raise MediaInspectionError(f"ffmpeg extraction failed for {media_path}: {exc.stderr.strip()}") from exc

    try:
        dur, _ = validate_wav_file(tmp_path, expected_sample_rate_hz=sample_rate_hz)
        os.replace(tmp_path, out_path)
        return dur
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
