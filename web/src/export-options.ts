import type { TaskKind } from "./api";

export const ARTIFACT_CHOICES = [
  { id: "source_transcript", label: "Source transcript (.txt)", tasks: ["transcription", "translation", "subtitles"] },
  { id: "final_english", label: "Final English dialogue (.txt)", tasks: ["translation", "subtitles"] },
  { id: "srt", label: "Timed subtitle track (.srt)", tasks: ["translation", "subtitles"] },
  { id: "ass", label: "Styled subtitle track (.ass)", tasks: ["subtitles"] },
  { id: "recording_contract", label: "ASR contract (.json)", tasks: ["transcription", "translation", "subtitles"] },
  { id: "refined_contract", label: "Source-preserving contract (.json)", tasks: ["translation", "subtitles"] },
  { id: "translated_contract", label: "Translation and review contract (.json)", tasks: ["translation", "subtitles"] },
  { id: "raw_source_transcript", label: "Original source transcript (.txt)", tasks: ["translation", "subtitles"] },
  { id: "refined_source_transcript", label: "Source compatibility transcript (.txt)", tasks: ["translation", "subtitles"] },
  { id: "subtitle_metadata", label: "Subtitle diagnostics (.json)", tasks: ["translation", "subtitles"] },
] as const;
export type ExportArtifactType = (typeof ARTIFACT_CHOICES)[number]["id"];
export function availableArtifacts(task: TaskKind): typeof ARTIFACT_CHOICES[number][] {
  return ARTIFACT_CHOICES.filter(entry => (entry.tasks as readonly string[]).includes(task));
}
export function selectedArtifacts(task: TaskKind, choices: readonly ExportArtifactType[]): ExportArtifactType[] {
  const valid = new Set<string>(availableArtifacts(task).map(entry => entry.id));
  return [...new Set(choices)].filter(item => valid.has(item));
}
