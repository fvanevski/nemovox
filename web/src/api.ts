export type MediaKind = "path" | "upload" | "url" | "contract";
export type TaskKind = "transcription" | "translation" | "subtitles";
export type Step = "media" | "configure" | "preview";

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
}
export interface JobOptions {
  source_language: string;
  target_language: string;
  asr_mode: "speaker" | "multispeaker_sot" | "rich" | "verbatim";
  audio_stream: number;
  chunk_seconds: number;
  context_seconds: number;
  speaker_embeddings: "titanet" | "none";
  alignment_fallback: "off" | "qwen";
  refinement: "off" | "conservative" | "balanced" | "expressive";
  translator: "swiftu" | "riva" | "none";
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
  glossaries: () => json<Glossary[]>("/glossaries"),
  jobs: () => json<Job[]>("/jobs"),
  job: (id: string) => json<Job>("/jobs/" + id),
  preview: (id: string) => json<Preview>("/jobs/" + id + "/preview"),
  artifacts: (id: string) => json<Artifact[]>("/jobs/" + id + "/artifacts"),
  logs: (id: string) => json<{lines: string[]}>("/jobs/" + id + "/logs"),
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
