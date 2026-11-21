"""Local-only Nemovox Studio HTTP API and single-worker CLI orchestration.

This module is deliberately separate from the ASR/runtime dependencies. It imports
no GPU model and executes the existing CLI rather than reimplementing its pipeline.
Run with one uvicorn worker bound to 127.0.0.1.
"""
from __future__ import annotations

import ipaddress
import json
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
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
    refinement: Literal["off", "conservative", "balanced", "expressive"] = "balanced"
    translator: Literal["swiftu", "riva", "none"] = "swiftu"
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
        # Jobs are durable enough for a single-user tool. Interrupted jobs can be resubmitted.
        for path in self.config.job_root.glob("*/job.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("state") in {"queued", "running"}:
                    record.update(state="interrupted", stage="Process stopped", updated_at=time.time())
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

    def cancel(self, job_id: str) -> dict:
        value = self.get(job_id)
        if value["state"] in {"complete", "failed", "cancelled", "interrupted"}:
            return value
        with self._lock:
            self._cancelled.add(job_id)
            process = self._processes.get(job_id)
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self.update(job_id, stage="Cancelling")
        return self.get(job_id)

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
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=8)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                        raise InterruptedError("Cancelled")
                    time.sleep(0.4)
                if job_id in self._cancelled:
                    raise InterruptedError("Cancelled")
                if process.returncode != 0:
                    raise RuntimeError(stage + " failed (exit " + str(process.returncode) + "); see execution.log")
            finally:
                with self._lock:
                    self._processes.pop(job_id, None)

    def _download(self, job_id: str, url: str) -> Path:
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
                        os.killpg(process.pid, signal.SIGTERM)
                        raise InterruptedError("Cancelled")
                    time.sleep(0.4)
                if process.returncode:
                    raise RuntimeError("Remote download failed; see execution.log")
            finally:
                with self._lock:
                    self._processes.pop(job_id, None)
        candidates = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS]
        if not candidates:
            raise RuntimeError("Downloader completed without a supported media file.")
        return max(candidates, key=lambda p: p.stat().st_size)

    def _run(self, job_id: str) -> None:
        try:
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

            if spec.source.kind == "contract":
                contract = source
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
            self.update(job_id, state="complete", stage="Complete", progress=100, finished_at=time.time())
        except InterruptedError:
            self.update(job_id, state="cancelled", stage="Cancelled", finished_at=time.time())
        except Exception as exc:
            self.update(job_id, state="failed", stage="Failed", error=str(exc), finished_at=time.time())


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
    app = FastAPI(title="Nemovox Studio", version="0.1.0")
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

    @app.get("/api/glossaries")
    def glossaries() -> list[dict]:
        root = PROJECT / "configs" / "glossaries"
        return [{"name": p.stem, "path": str(p)} for p in sorted(root.glob("*"))
                if p.suffix.lower() in {".yaml", ".yml", ".json"} and p.is_file()]

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
            elif source.suffix.lower() not in MEDIA_EXTENSIONS:
                raise HTTPException(422, "Unsupported media format.")
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
            "utterance_id": u.get("utterance_id"),
        } for u in value.get("utterances", [])]
        return {"kind": "transcript", "cues": cues}

    @app.get("/api/jobs/{job_id}/artifacts")
    def artifacts(job_id: str) -> list[dict]:
        root = runner.path(job_id)
        paths = [
            root / "asr/recording.contract.json",
            root / "output/refined.contract.json",
            root / "output/translated.contract.json",
            root / "output/final_english_subtitles.txt",
            *sorted((root / "output").glob("*.srt")),
            *sorted((root / "output").glob("*.ass")),
        ]
        return [{
            "name": p.name, "path": str(p.relative_to(root)),
            "url": "/api/jobs/" + job_id + "/artifact/" + p.relative_to(root).as_posix(),
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
        if not path.is_file() or not _within(path, allowed) or path.suffix.lower() not in MEDIA_EXTENSIONS:
            raise HTTPException(404, "Media not available for preview.")
        return FileResponse(path)

    frontend = PROJECT / "web" / "dist"
    if frontend.is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=frontend, html=True), name="web")
    return app


app = create_app()
