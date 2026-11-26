export type MediaKind = "path" | "upload" | "url" | "contract";
export type TaskKind = "transcription" | "translation" | "subtitles";
export type Step = "media" | "configure" | "preview";
import type { ExportArtifactType } from "./export-options";
export type { ExportArtifactType } from "./export-options";

export interface ExportPlan {
  destination: string | null;
  artifacts: ExportArtifactType[];
  save_remote_media: boolean;
}
export interface ExportReport {
  status: "disabled" | "pending" | "complete" | "partial" | "failed";
  directory: string | null;
  items: {type: string; status: string; source?: string; destination?: string; error?: string; sha256?: string}[];
  updated_at?: number;
}
export interface StageEvent {
  id: string;
  status: "running" | "complete" | "skipped" | "failed" | "cancelled";
  at: number;
  detail: string;
}
export interface ExportDirectoryListing {
  roots: string[];
  default: string;
  directory: string;
  parent: string | null;
  directories: {name: string; path: string}[];
}

export interface MediaEntry {
  name: string;
  path: string;
  kind: "directory" | "file";
}
export interface MediaListing {
  directory: string | null;
  parent?: string | null;
  items: MediaEntry[];
}
export interface Glossary {
  name: string;
  path: string;
  characters: string[];
  terms: string[];
  notes: string;
}
export interface AudioStream {
  index: number;
  container_index: number | null;
  codec: string | null;
  language: string | null;
  title: string | null;
  channels: number | null;
  sample_rate_hz: number | null;
}
export interface ModelReadiness {
  state: string;
  ready: boolean;
  managed?: boolean;
  startable: boolean;
}
export interface ManagedModelStatus {
  asr: ModelReadiness;
  translation_review: ModelReadiness;
  error?: string;
}
export interface TranslationBackendStatus {
  riva: {configured: boolean; device: string | null; detail: string};
}
export interface JobOptions {
  source_language: string;
  target_language: string;
  audio_stream: number;
  refinement: "off" | "conservative" | "balanced" | "expressive";
  translator: "openai_chat" | "riva" | "none";
  translation_review: "off" | "conservative" | "balanced" | "expressive";
  style: "natural" | "literal" | "expressive";
  glossary: string | null;
  ass_layout: "standard" | "raised_main";
  ass_max_lanes: 2 | 3;
  condense_high_cps: boolean;
  formats: ("srt" | "ass")[];
}
export interface JobRequest {
  source: { kind: MediaKind; value: string };
  task: TaskKind;
  options: JobOptions;
  export?: ExportPlan;
}
export interface Job {
  id: string;
  state: "queued" | "running" | "complete" | "failed" | "interrupted" | "cancelled";
  stage: string;
  progress: number;
  request: JobRequest;
  source_path: string | null;
  created_at: number;
  updated_at: number;
  error: string | null;
  export_report?: ExportReport;
  stage_history?: StageEvent[];
}
export interface Cue {
  start: number | null;
  end: number | null;
  text: string;
  speaker?: string | null;
  utterance_id?: string | null;
}
export interface Preview { kind: "pending" | "subtitles" | "transcript"; cues: Cue[]; }
export interface Artifact { name: string; path: string; url: string; }

async function json<T>(route: string, init?: RequestInit): Promise<T> {
  const res = await fetch("/api" + route, init);
  if (!res.ok) {
    let message = "Request failed (" + res.status + ")";
    try {
      const body = await res.json();
      if (typeof body.detail === "string") message = body.detail;
      else if (typeof body.error === "string") message = body.error;
    } catch { /* non-JSON response */ }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}
export const studio = {
  health: () => json<{status: string; ffmpeg: boolean; ytdlp: boolean; media_roots: string[]}>("/health"),
  browse: (path?: string | null) => json<MediaListing>("/media" + (path ? "?path=" + encodeURIComponent(path) : "")),
  exportDirectories: (path?: string | null) => json<ExportDirectoryListing>(
    "/export-directories" + (path ? "?path=" + encodeURIComponent(path) : "")),
  history: (offset = 0, limit = 20) => json<{items: Job[]; total: number; next_offset: number | null}>(
    "/job-history?offset=" + offset + "&limit=" + limit),
  glossaries: () => json<Glossary[]>("/glossaries"),
  inspect: (source: JobRequest["source"]) => json<{audio_streams: AudioStream[]}>("/media/inspect", {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(source),
  }),
  translationBackends: () => json<TranslationBackendStatus>("/translation-backends"),
  modelStatus: () => json<ManagedModelStatus>("/resource-status"),
  jobs: () => json<Job[]>("/jobs"),
  job: (id: string) => json<Job>("/jobs/" + id),
  preview: (id: string) => json<Preview>("/jobs/" + id + "/preview"),
  artifacts: (id: string) => json<Artifact[]>("/jobs/" + id + "/artifacts"),
  logs: (id: string) => json<{lines: string[]}>("/jobs/" + id + "/logs"),
  logChunk: (id: string, offset: number, limit = 65536) => json<{
    text: string; next_offset: number; total: number; complete: boolean;
  }>("/jobs/" + id + "/log/chunk?offset=" + offset + "&limit=" + limit),
  logDownload: (id: string) => "/api/jobs/" + id + "/log/download",
  exports: (id: string) => json<ExportReport>("/jobs/" + id + "/exports"),
  submit: (body: JobRequest) => json<Job>("/jobs", {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body),
  }),
  cancel: (id: string) => json<Job>("/jobs/" + id + "/cancel", {method: "POST"}),
  upload: async (file: File) => {
    const body = new FormData();
    body.append("media_file", file);
    return json<{path: string; size_bytes: number}>("/media/upload", {method: "POST", body});
  },
};
export function secondsLabel(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const h = Math.floor(value / 3600);
  const m = Math.floor((value % 3600) / 60);
  const s = Math.floor(value % 60);
  const decimals = Math.round((value - Math.floor(value)) * 100);
  return (h ? String(h).padStart(2, "0") + ":" : "") +
    String(m).padStart(2, "0") + ":" + String(s).padStart(2, "0") +
    "." + String(decimals).padStart(2, "0");
}
