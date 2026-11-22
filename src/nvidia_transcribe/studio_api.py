"""Local-only Nemovox Studio HTTP API and single-worker CLI orchestration.

This module is deliberately separate from the ASR/runtime dependencies. It imports
no GPU model and executes the existing CLI rather than reimplementing its pipeline.
Run with one uvicorn worker bound to 127.0.0.1.
"""
from __future__ import annotations

import fcntl
import ipaddress
import json
import math
import os
import re
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import quote, urlsplit

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from nvidia_transcribe.media import MediaInspectionError, inspect_audio_streams
from nvidia_transcribe.model_resources import ResourceConfigError, load_model_resources
from nvidia_transcribe.refinement import load_glossary
from pydantic import BaseModel, Field

PROJECT = Path(__file__).resolve().parents[2]
VIDEO_EXTENSIONS = {".webm", ".mp4", ".mkv", ".mov", ".avi", ".flv", ".m4v"}
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac"}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS
JOB_ID = re.compile(r"^[a-f0-9]{32}$")
MAX_UPLOAD_BYTES = 4 * 1024 ** 3


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _within(path: Path, roots: tuple[Path, ...]) -> bool:
    real = path.resolve()
    return any(real.is_relative_to(root.resolve()) for root in roots)


def _remote_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Use a public HTTP(S) media URL without embedded credentials.")
    host = parsed.hostname.lower()
    if host == "localhost" or host.endswith(".local") or host.endswith(".localhost"):
        raise ValueError("Local/private network URLs are not supported.")
    try:
        addresses = socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
        if not addresses:
            raise ValueError("Could not resolve media host.")
        for entry in addresses:
            ip = ipaddress.ip_address(entry[4][0])
            if not ip.is_global:
                raise ValueError("Remote URL resolves to a non-public address.")
    except socket.gaierror as exc:
        raise ValueError("Could not resolve media host.") from exc
    return url


def _read_transcription_contract(path: Path) -> dict:
    """Validate source utterances before importing a prior recording transcription."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Recording contract is not readable JSON.") from exc
    if not isinstance(value, dict) or not isinstance(value.get("utterances"), list):
        raise ValueError("Recording contract must contain an utterances array.")
    if any(not isinstance(item, dict) or not isinstance(item.get("source_text"), str)
           for item in value["utterances"]):
        raise ValueError("Every contract utterance must contain source_text.")
    return value


def _format_existing_transcript(value: dict) -> str:
    """Format existing source-language content without changing text or inventing times."""
    source = value.get("source")
    source_path = source.get("path", "existing contract") if isinstance(source, dict) else "existing contract"
    language = value.get("source_language") or "und"
    provenance = value.get("language_provenance") or "unspecified"
    utterances = value["utterances"]
    lines = [f"# Source: {source_path} | Language: {language} ({provenance}) | Utterances: {len(utterances)}", ""]
    if not utterances:
        lines.append("[No utterances in source contract]")
    for item in utterances:
        start, end = item.get("source_start"), item.get("source_end")
        if all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t)
               for t in (start, end)):
            span = f"[{start:07.2f} -> {end:07.2f}]"
        else:
            span = "[untimed -> untimed]"
        speaker = item.get("speaker_id") or "unknown"
        status = item.get("alignment_status") or "unknown"
        lines.append(f"{span} {speaker} ({status}): {item['source_text']}")
    return "\n".join(lines) + "\n"


class Source(BaseModel):
    kind: Literal["path", "upload", "url", "contract"]
    value: str = Field(min_length=1, max_length=4096)


class Options(BaseModel):
    source_language: str = Field(default="auto", max_length=24)
    target_language: str = Field(default="en", max_length=24)
    asr_mode: Literal["speaker", "multispeaker_sot", "verbatim", "rich"] = "speaker"
    audio_stream: int = Field(default=0, ge=0, le=15)
    chunk_seconds: int = Field(default=90, ge=30, le=300)
    context_seconds: int = Field(default=10, ge=0, le=30)
    speaker_embeddings: Literal["titanet", "none"] = "titanet"
    alignment_fallback: Literal["off", "qwen"] = "off"
    refinement: Literal["off", "conservative", "balanced", "expressive"] = "off"
    translator: Literal["openai_chat", "swiftu", "riva", "none"] = "openai_chat"
    translation_review: Literal["off", "conservative", "balanced", "expressive"] = "off"
    style: Literal["natural", "literal", "expressive"] = "natural"
    glossary: str | None = Field(default=None, max_length=512)
    ass_layout: Literal["standard", "raised_main"] = "standard"
    ass_max_lanes: Literal[2, 3] = 3
    condense_high_cps: bool = True
    formats: list[Literal["srt", "ass"]] = Field(default_factory=lambda: ["srt", "ass"])


class JobRequest(BaseModel):
    source: Source
    task: Literal["transcription", "translation", "subtitles"] = "subtitles"
    options: Options = Field(default_factory=Options)


@dataclass(frozen=True)
class StudioConfig:
    job_root: Path
    upload_root: Path
    media_roots: tuple[Path, ...]
    python: str = sys.executable

    @classmethod
    def from_env(cls) -> "StudioConfig":
        roots = [
            PROJECT / "outputs",
            Path.home() / "Videos",
            Path("/opt/mcog"),
            Path("/workspace/pyannote/outputs/sources"),
        ]
        extra = os.environ.get("NEMOVOX_MEDIA_ROOTS", "")
        roots.extend(Path(s) for s in extra.split(os.pathsep) if s)
        existing = tuple(p.resolve() for p in roots if p.is_dir())
        return cls(
            job_root=Path(os.getenv("NEMOVOX_STUDIO_JOBS", PROJECT / "outputs/studio/jobs")).resolve(),
            upload_root=Path(os.getenv("NEMOVOX_STUDIO_UPLOADS", PROJECT / "outputs/studio/uploads")).resolve(),
            media_roots=existing,
            python=os.environ.get("NEMOVOX_CLI_PYTHON", str(PROJECT / ".venv/bin/python") if (PROJECT / ".venv/bin/python").is_file() else sys.executable),
        )


class JobRunner:
    def __init__(self, config: StudioConfig):
        self.config = config
        self.config.job_root.mkdir(parents=True, exist_ok=True)
        self.config.upload_root.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nemovox-studio")
        self._lock = threading.Lock()
        self._processes: dict[str, subprocess.Popen] = {}
        self._cancelled: set[str] = set()
        # Creating an app/importing this module must never rewrite jobs owned by
        # a running Studio instance. Reconciliation happens only after startup
        # acquires the exclusive job-root owner lock.

    @contextmanager
    def owned_job_root(self):
        """Prevent a second Studio process from adopting another live job queue."""
        lock_file = self.config.job_root / ".studio-owner.lock"
        with lock_file.open("a+b") as owner:
            try:
                fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    "Studio job directory is already owned by another running server."
                ) from exc
            try:
                self._reconcile_previous_jobs()
                yield
            finally:
                fcntl.flock(owner.fileno(), fcntl.LOCK_UN)

    def _reconcile_previous_jobs(self) -> None:
        """Only the admitted startup owner may classify previous incomplete jobs."""
        for path in self.config.job_root.glob("*/job.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("state") in {"queued", "running"}:
                    record.update(
                        state="interrupted", stage="Previous Studio process stopped",
                        error="Prior subprocess cleanup cannot be verified after an unclean restart.",
                        updated_at=time.time(),
                    )
                    _atomic_json(path, record)
            except (OSError, ValueError):
                continue

    def path(self, job_id: str) -> Path:
        if not JOB_ID.fullmatch(job_id):
            raise HTTPException(404, "Job not found.")
        dest = self.config.job_root / job_id
        if not (dest / "job.json").is_file():
            raise HTTPException(404, "Job not found.")
        return dest

    def get(self, job_id: str) -> dict:
        return json.loads((self.path(job_id) / "job.json").read_text(encoding="utf-8"))

    def update(self, job_id: str, **changes: object) -> None:
        with self._lock:
            path = self.config.job_root / job_id / "job.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value.update(changes, updated_at=time.time())
            _atomic_json(path, value)

    def submit(self, request: JobRequest, source_path: Path | None) -> dict:
        job_id = uuid.uuid4().hex
        dest = self.config.job_root / job_id
        dest.mkdir(parents=True)
        record = {
            "id": job_id,
            "state": "queued",
            "stage": "Queued",
            "progress": 0,
            "request": request.model_dump(),
            "source_path": str(source_path) if source_path else None,
            "created_at": time.time(),
            "updated_at": time.time(),
            "error": None,
        }
        _atomic_json(dest / "job.json", record)
        self.executor.submit(self._run, job_id)
        return record

    @staticmethod
    def _group_has_live_members(pgid: int) -> bool:
        """Inspect the owned session even after its Popen group leader has exited.

        Linux /proc distinguishes live descendants from unreaped zombies.
        """
        proc = Path("/proc")
        if proc.is_dir():
            for entry in proc.iterdir():
                if not entry.name.isdecimal():
                    continue
                try:
                    # /proc/PID/stat after comm: state, ppid, pgrp, session, ...
                    fields = (entry / "stat").read_text(encoding="utf-8").rsplit(") ", 1)[1].split()
                    if (fields[0] not in {"Z", "X"} and int(fields[2]) == pgid
                            and int(fields[3]) == pgid):
                        return True
                except (OSError, ValueError, IndexError):
                    continue
            return False
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        return True

    @classmethod
    def _terminate_process_group(cls, process: subprocess.Popen) -> None:
        """Bounded TERM/KILL of owned session members, regardless of leader status."""
        pgid = process.pid  # start_new_session=True => PGID == SID == PID.
        for sig, grace_seconds in ((signal.SIGTERM, 2.0), (signal.SIGKILL, 2.0)):
            if not cls._group_has_live_members(pgid):
                break
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                break
            deadline = time.monotonic() + grace_seconds
            while cls._group_has_live_members(pgid) and time.monotonic() < deadline:
                time.sleep(0.05)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Owned subprocess leader did not terminate.") from exc
        if cls._group_has_live_members(pgid):
            raise RuntimeError("Owned subprocess descendants remain after SIGKILL.")

    def shutdown(self) -> None:
        """Stop active jobs on graceful Studio shutdown; do not claim crash cleanup."""
        for path in self.config.job_root.glob("*/job.json"):
            try:
                if self.get(path.parent.name)["state"] in {"queued", "running"}:
                    self.cancel(path.parent.name)
            except (OSError, ValueError, HTTPException):
                continue
        self.executor.shutdown(wait=True, cancel_futures=False)
        for path in self.config.job_root.glob("*/job.json"):
            try:
                if self.get(path.parent.name)["state"] in {"queued", "running"}:
                    self.update(
                        path.parent.name, state="interrupted", stage="Service stopped",
                        error="Job stopped before completion.", finished_at=time.time(),
                    )
            except (OSError, ValueError, HTTPException):
                continue

    def cancel(self, job_id: str) -> dict:
        # Cancellation and successful finalization must observe one serialized
        # terminal-state boundary. Otherwise a late cancel can be overwritten
        # by _run's unconditional complete update.
        with self._lock:
            path = self.path(job_id) / "job.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            if value["state"] in {"complete", "failed", "cancelled", "interrupted"}:
                return value
            self._cancelled.add(job_id)
            process = self._processes.get(job_id)
            value.update(stage="Cancelling", updated_at=time.time())
            _atomic_json(path, value)
        if process is not None and self._group_has_live_members(process.pid):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        return self.get(job_id)

    def _complete_if_not_cancelled(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._cancelled:
                raise InterruptedError("Cancelled")
            path = self.path(job_id) / "job.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            if value["state"] != "running":
                raise RuntimeError(f"Cannot complete Studio job from state {value['state']!r}.")
            value.update(
                state="complete", stage="Complete", progress=100,
                finished_at=time.time(), updated_at=time.time(),
            )
            _atomic_json(path, value)

    def _command(self, job_id: str, args: list[str], stage: str, progress: int) -> None:
        if job_id in self._cancelled:
            raise InterruptedError("Cancelled")
        self.update(job_id, stage=stage, progress=progress)
        path = self.config.job_root / job_id / "execution.log"
        command = [self.config.python, "-m", "nvidia_transcribe.cli", *args]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        with path.open("a", encoding="utf-8") as log:
            log.write("\nSTAGE: " + stage + "\n")
            log.flush()
            process = subprocess.Popen(
                command, cwd=PROJECT, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            with self._lock:
                self._processes[job_id] = process
            try:
                while process.poll() is None:
                    if job_id in self._cancelled:
                        raise InterruptedError("Cancelled")
                    time.sleep(0.4)
                if job_id in self._cancelled:
                    raise InterruptedError("Cancelled")
                if process.returncode != 0:
                    raise RuntimeError(stage + " failed (exit " + str(process.returncode) + "); see execution.log")
            finally:
                try:
                    self._terminate_process_group(process)
                finally:
                    with self._lock:
                        self._processes.pop(job_id, None)

    def _download(self, job_id: str, url: str) -> Path:
        if job_id in self._cancelled:
            raise InterruptedError("Cancelled")
        if shutil.which("yt-dlp") is None:
            raise RuntimeError("Install yt-dlp to process remote media URLs.")
        root = self.config.job_root / job_id / "remote"
        root.mkdir(exist_ok=True)
        self.update(job_id, stage="Downloading remote media", progress=4)
        target = root / "source.%(ext)s"
        cmd = [
            "yt-dlp", "--no-playlist", "--no-progress", "--restrict-filenames",
            "--max-filesize", "4G", "-o", str(target), url,
        ]
        log_path = self.config.job_root / job_id / "execution.log"
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                cmd, cwd=root, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            with self._lock:
                self._processes[job_id] = process
            try:
                while process.poll() is None:
                    if job_id in self._cancelled:
                        raise InterruptedError("Cancelled")
                    time.sleep(0.4)
                if job_id in self._cancelled:
                    raise InterruptedError("Cancelled")
                if process.returncode:
                    raise RuntimeError("Remote download failed; see execution.log")
            finally:
                try:
                    self._terminate_process_group(process)
                finally:
                    with self._lock:
                        self._processes.pop(job_id, None)
        candidates = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS]
        if not candidates:
            raise RuntimeError("Downloader completed without a supported media file.")
        return max(candidates, key=lambda p: p.stat().st_size)

    def _run(self, job_id: str) -> None:
        try:
            if job_id in self._cancelled:
                raise InterruptedError("Cancelled")
            self.update(job_id, state="running", stage="Preparing media", progress=2)
            value = self.get(job_id)
            spec = JobRequest.model_validate(value["request"])
            opt = spec.options
            root = self.config.job_root / job_id
            if spec.source.kind == "url":
                source = self._download(job_id, spec.source.value)
            else:
                source = Path(value["source_path"])
            self.update(job_id, source_path=str(source))

            # The CLI resolves NEMOVOX_MODEL_RESOURCES and text/ASR env in its
            # child environment. Studio must not elevate env to CLI overrides.

            if spec.source.kind == "contract":
                contract = source
                if spec.task == "transcription":
                    value = _read_transcription_contract(source)
                    imported = root / "asr"
                    imported.mkdir(parents=True, exist_ok=True)
                    _atomic_json(imported / "recording.contract.json", value)
                    (imported / "transcript.txt").write_text(
                        _format_existing_transcript(value), encoding="utf-8"
                    )
                    contract = imported / "recording.contract.json"
            else:
                args = [
                    "process-media", str(source), "--output-dir", str(root / "asr"),
                    "--mode", opt.asr_mode, "--audio-stream", str(opt.audio_stream),
                    "--chunk-seconds", str(opt.chunk_seconds),
                    "--context-seconds", str(opt.context_seconds),
                    "--speaker-embeddings", opt.speaker_embeddings,
                    "--alignment-fallback", opt.alignment_fallback,
                ]
                if opt.source_language != "auto":
                    args.extend(["--source-language", opt.source_language])
                self._command(job_id, args, "Transcribing audio and speakers", 8)
                contract = root / "asr" / "recording.contract.json"
            if spec.task != "transcription":
                args = [
                    "generate-subtitles", "--recording-contract", str(contract),
                    "--output-dir", str(root / "output"),
                    "--target-language", opt.target_language,
                    "--refinement", opt.refinement, "--translator", opt.translator,
                    "--translation-review", opt.translation_review,
                    "--style", opt.style, "--ass-layout", opt.ass_layout,
                    "--ass-max-lanes", str(opt.ass_max_lanes),
                    "--formats", ",".join(opt.formats if spec.task == "subtitles" else ["srt"]),
                ]
                if opt.source_language != "auto":
                    args.extend(["--source-language", opt.source_language])
                if opt.condense_high_cps and spec.task == "subtitles":
                    args.append("--condense-high-cps")
                if opt.glossary:
                    args.extend(["--glossary", opt.glossary])
                self._command(job_id, args, "Refining, translating and rendering", 60)
            self._complete_if_not_cancelled(job_id)
        except InterruptedError:
            self.update(job_id, state="cancelled", stage="Cancelled", finished_at=time.time())
        except Exception as exc:
            self.update(job_id, state="failed", stage="Failed", error=str(exc), finished_at=time.time())
        finally:
            with self._lock:
                self._cancelled.discard(job_id)


def _srt_cues(text: str) -> list[dict]:
    stamp = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")
    def seconds(match: re.Match) -> float:
        h, m, s, ms = map(int, match.groups())
        return h * 3600 + m * 60 + s + ms / 1000
    cues = []
    for item in re.split(r"\r?\n\s*\r?\n", text.strip()):
        lines = item.splitlines()
        for index, line in enumerate(lines):
            if "-->" not in line:
                continue
            start, end = line.split("-->", 1)
            a, b = stamp.search(start), stamp.search(end)
            if a and b:
                cues.append({"start": seconds(a), "end": seconds(b), "text": "\n".join(lines[index + 1:])})
            break
    return cues


def create_app(config: StudioConfig | None = None) -> FastAPI:
    settings = config or StudioConfig.from_env()
    runner = JobRunner(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        with runner.owned_job_root():
            try:
                yield
            finally:
                runner.shutdown()

    app = FastAPI(title="Nemovox Studio", version="0.1.0", lifespan=lifespan)
    app.state.runner = runner
    app.add_middleware(
        CORSMiddleware, allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
    )

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok", "backend": "local",
            "ffmpeg": shutil.which("ffmpeg") is not None,
            "ytdlp": shutil.which("yt-dlp") is not None,
            "media_roots": [str(p) for p in settings.media_roots],
        }

    @app.get("/api/media")
    def media(path: str | None = Query(default=None)) -> dict:
        if not path:
            return {"directory": None, "items": [
                {"name": p.name or str(p), "path": str(p), "kind": "directory"}
                for p in settings.media_roots
            ]}
        candidate = Path(path).expanduser().resolve()
        if not _within(candidate, settings.media_roots) or not candidate.is_dir():
            raise HTTPException(403, "Directory is not inside an allowed media root.")
        items = []
        for item in sorted(candidate.iterdir(), key=lambda p: (p.is_file(), p.name.casefold()))[:250]:
            if item.is_symlink() and not _within(item, settings.media_roots):
                continue
            if item.is_dir() or item.suffix.lower() in MEDIA_EXTENSIONS or item.name == "recording.contract.json":
                items.append({
                    "name": item.name, "path": str(item),
                    "kind": "directory" if item.is_dir() else "file",
                })
        return {"directory": str(candidate), "parent": str(candidate.parent) if _within(candidate.parent, settings.media_roots) else None, "items": items}

    @app.post("/api/media/inspect")
    def inspect_media(source: Source) -> dict:
        if source.kind not in ("path", "upload"):
            raise HTTPException(422, "Only selected local or uploaded media can be inspected before processing.")
        path = Path(source.value).expanduser().resolve()
        roots = settings.media_roots if source.kind == "path" else (settings.upload_root,)
        if not _within(path, roots) or not path.is_file():
            raise HTTPException(403, "Selected media is outside allowed local roots.")
        if path.suffix.lower() not in MEDIA_EXTENSIONS:
            raise HTTPException(422, "Unsupported media format.")
        try:
            tracks = inspect_audio_streams(path)
        except MediaInspectionError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"audio_streams": tracks}

    @app.get("/api/glossaries")
    def glossaries() -> list[dict]:
        root = PROJECT / "configs" / "glossaries"
        records = []
        for p in sorted(root.glob("*")):
            if p.suffix.lower() not in {".yaml", ".yml", ".json"} or not p.is_file():
                continue
            try:
                glossary = load_glossary(p)
                if p.suffix.lower() in (".yaml", ".yml"):
                    import yaml
                    metadata = yaml.safe_load(p.read_text(encoding="utf-8"))
                else:
                    metadata = json.loads(p.read_text(encoding="utf-8"))
                title = metadata.get("title") if isinstance(metadata, dict) else None
                records.append({
                    "name": str(title).strip()[:120] if title else p.stem.replace("_", " "),
                    "path": str(p),
                    "characters": list(glossary.characters)[:40],
                    "terms": list(glossary.terms)[:60],
                    "notes": (glossary.notes or "")[:1200],
                })
            except (OSError, TypeError, ValueError):
                continue
        return records

    @app.get("/api/translation-backends")
    def translation_backends() -> dict:
        # Configuration discovery only. Do not load models or probe GPU/services.
        try:
            resource = load_model_resources().resolve("riva_translation")
            checkpoint = Path(resource.checkpoint_path or "")
            worker = Path(resource.worker_python or "")
            installed = checkpoint.is_dir() and worker.is_file()
            return {
                "riva": {
                    "configured": installed,
                    "device": resource.device or "cuda:0",
                    "detail": ("Dedicated Riva translation worker is configured." if installed
                               else "Riva checkpoint or worker interpreter is unavailable on this host."),
                }
            }
        except (ResourceConfigError, OSError):
            return {"riva": {"configured": False, "device": None,
                             "detail": "Riva resource configuration is unavailable."}}

    @app.post("/api/media/upload")
    async def upload(media_file: UploadFile = File(...)) -> dict:
        suffix = Path(media_file.filename or "").suffix.lower()
        if suffix not in MEDIA_EXTENSIONS:
            raise HTTPException(422, "Unsupported media extension.")
        settings.upload_root.mkdir(parents=True, exist_ok=True)
        path = settings.upload_root / (uuid.uuid4().hex + suffix)
        size = 0
        try:
            with path.open("wb") as fp:
                while chunk := await media_file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "Media upload exceeds 4 GiB.")
                    fp.write(chunk)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            await media_file.close()
        return {"path": str(path), "size_bytes": size}

    @app.post("/api/jobs", status_code=202)
    def submit(request: JobRequest) -> dict:
        if request.task != "transcription" and request.options.translator == "none":
            raise HTTPException(422, "Select a translation backend for this task.")
        if request.task != "transcription" and request.options.refinement != "off":
            raise HTTPException(
                422, "Studio uses the original ASR transcript without source-language correction. "
                     "For legacy source correction, explicitly use CLI --refinement.",
            )
        if (request.task != "transcription" and request.options.translation_review != "off"
                and request.options.translator != "riva"):
            raise HTTPException(422, "English post-edit requires dedicated Riva translation.")
        if request.options.glossary:
            glossary = Path(request.options.glossary).resolve()
            if not _within(glossary, (PROJECT / "configs" / "glossaries",)) or not glossary.is_file():
                raise HTTPException(422, "Glossary must exist in configs/glossaries.")
        if request.source.kind == "url":
            try:
                _remote_url(request.source.value)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            source = None
        else:
            source = Path(request.source.value).expanduser().resolve()
            roots = settings.media_roots + (settings.upload_root.resolve(),)
            if not source.is_file() or not _within(source, roots):
                raise HTTPException(403, "Selected media is outside allowed local roots.")
            if request.source.kind == "contract":
                if source.name != "recording.contract.json":
                    raise HTTPException(422, "Select a recording.contract.json.")
                if request.task == "transcription":
                    try:
                        _read_transcription_contract(source)
                    except ValueError as exc:
                        raise HTTPException(422, str(exc)) from exc
            elif source.suffix.lower() not in MEDIA_EXTENSIONS:
                raise HTTPException(422, "Unsupported media format.")
            else:
                try:
                    tracks = inspect_audio_streams(source)
                except MediaInspectionError as exc:
                    raise HTTPException(422, str(exc)) from exc
                if request.options.audio_stream >= len(tracks):
                    raise HTTPException(422, "Selected audio track is not present in this file.")
        if request.source.kind == "url" and request.options.audio_stream != 0:
            raise HTTPException(422, "Remote audio tracks are unknown until download; use track 0.")
        if request.task != "transcription" and request.options.translator == "riva":
            try:
                load_model_resources().resolve("riva_translation", require_local_paths=True)
            except ResourceConfigError as exc:
                raise HTTPException(422, f"Dedicated Riva translation is not available: {exc}") from exc
        return runner.submit(request, source)

    @app.get("/api/jobs")
    def jobs() -> list[dict]:
        records = []
        for path in settings.job_root.glob("*/job.json"):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return sorted(records, key=lambda r: r["created_at"], reverse=True)[:50]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict:
        return runner.get(job_id)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> dict:
        return runner.cancel(job_id)

    @app.get("/api/jobs/{job_id}/logs")
    def logs(job_id: str) -> dict:
        path = runner.path(job_id) / "execution.log"
        if not path.exists():
            return {"lines": []}
        with path.open("rb") as fp:
            fp.seek(max(0, path.stat().st_size - 24000))
            value = fp.read().decode("utf-8", errors="replace")
        return {"lines": value.splitlines()[-100:]}

    @app.get("/api/jobs/{job_id}/preview")
    def preview(job_id: str) -> dict:
        root = runner.path(job_id)
        output = root / "output"
        subs = sorted(output.glob("*.srt"))
        if subs:
            return {"kind": "subtitles", "cues": _srt_cues(subs[0].read_text(encoding="utf-8"))}
        candidate = output / "translated.contract.json"
        if not candidate.is_file():
            candidate = root / "asr" / "recording.contract.json"
        if not candidate.is_file():
            return {"kind": "pending", "cues": []}
        value = json.loads(candidate.read_text(encoding="utf-8"))
        cues = [{
            "start": u.get("source_start"), "end": u.get("source_end"),
            "text": u.get("translated_text") or u.get("refined_text") or u.get("source_text") or "",
            "speaker": u.get("speaker_id") or u.get("global_speaker_id"),
            "utterance_id": u.get("utterance_id") or u.get("id"),
        } for u in value.get("utterances", [])]
        return {"kind": "transcript", "cues": cues}

    @app.get("/api/jobs/{job_id}/artifacts")
    def artifacts(job_id: str) -> list[dict]:
        root = runner.path(job_id)
        paths = [
            root / "asr/recording.contract.json",
            root / "asr/transcript.txt",
            root / "output/refined.contract.json",
            root / "output/translated.contract.json",
            root / "output/raw_source_transcript.txt",
            root / "output/refined_source_transcript.txt",
            root / "output/final_english_subtitles.txt",
            root / "output/subtitles.metadata.json",
            *sorted((root / "output").glob("*.srt")),
            *sorted((root / "output").glob("*.ass")),
        ]
        return [{
            "name": p.name, "path": str(p.relative_to(root)),
            "url": "/api/jobs/" + job_id + "/artifact/" + quote(p.relative_to(root).as_posix(), safe="/"),
        } for p in paths if p.is_file()]

    @app.get("/api/jobs/{job_id}/artifact/{path:path}")
    def artifact(job_id: str, path: str):
        root = runner.path(job_id)
        candidate = (root / path).resolve()
        if not candidate.is_file() or not _within(candidate, (root / "asr", root / "output")):
            raise HTTPException(404, "Artifact not found.")
        if candidate.suffix not in {".json", ".txt", ".srt", ".ass"}:
            raise HTTPException(403, "Artifact type not permitted.")
        return FileResponse(candidate, filename=candidate.name)

    @app.get("/api/jobs/{job_id}/media")
    def stream_media(job_id: str):
        value = runner.get(job_id)
        path_raw = value.get("source_path")
        if not path_raw:
            raise HTTPException(404, "Media not available yet.")
        path = Path(path_raw).resolve()
        allowed = settings.media_roots + (settings.upload_root, settings.job_root)
        if path.suffix.lower() == ".json" and path.is_file() and _within(path, allowed):
            try:
                contract_doc = json.loads(path.read_text(encoding="utf-8"))
                embedded = (contract_doc.get("source") or {}).get("path")
                if embedded:
                    path = Path(str(embedded)).expanduser().resolve()
            except (OSError, ValueError):
                raise HTTPException(404, "Media not available for preview.")
        if not path.is_file() or not _within(path, allowed) or path.suffix.lower() not in MEDIA_EXTENSIONS:
            raise HTTPException(404, "Media not available for preview.")
        return FileResponse(path)

    frontend = PROJECT / "web" / "dist"
    if frontend.is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=frontend, html=True), name="web")
    return app


app = create_app()
