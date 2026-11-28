import type { AudioStream, JobOptions } from "./api";

/** English-source subtitles need no MT or source-language text correction. */
export function isEnglishSourceToEnglish(source: string, target: string): boolean {
  const base = (language: string) => language.trim().toLowerCase().replaceAll("_", "-").split("-")[0];
  return ["en", "eng", "english"].includes(base(source)) &&
    ["en", "eng", "english"].includes(base(target));
}

export const INITIAL_OPTIONS: JobOptions = {
  source_language: "auto", target_language: "en", audio_stream: 0,
  refinement: "off", translator: "riva", translation_review: "balanced", style: "natural",
  glossary: null, ass_layout: "auto", ass_max_lanes: 3,
  formats: ["srt", "ass"],
};

export interface ExplainedOption {
  value: string;
  label: string;
  description: string;
}

export const TRANSLATION_REVIEW: readonly ExplainedOption[] = [
  {value: "off", label: "Off: raw machine translation",
   description: "Keep the dedicated translation model's English output as-is. Machine translation remains available for inspection."},
  {value: "conservative", label: "Conservative",
   description: "Correct clear translation errors, misrendered names and obvious inconsistencies against source dialogue."},
  {value: "balanced", label: "Balanced (recommended)",
   description: "Review coherent source passages against primary machine translation, with comparative checks of changed cues and joined sentences. Retain primary translation when support is insufficient."},
  {value: "expressive", label: "Expressive",
   description: "Where an actual translation problem is identified, consider tone and naturalness without rewriting otherwise acceptable cues or inventing details."},
];
export const LOCALIZATION_STYLES: readonly ExplainedOption[] = [
  {value: "natural", label: "Natural dialogue",
   description: "Favors concise, idiomatic and watchable English while keeping the original meaning."},
  {value: "literal", label: "Closer to literal",
   description: "Stays closer to the source phrasing and details, with fewer stylistic paraphrases."},
  {value: "expressive", label: "Expressive localization",
   description: "Emphasizes the original emotional tone and colloquialisms without introducing new story facts."},
];

export function normalizeAudioIndex(tracks: readonly AudioStream[] | null | undefined, current: number): number {
  return tracks?.some(t => t.index === current) ? current : 0;
}

export function hasAudioIndex(tracks: readonly AudioStream[] | null | undefined, selected: number): boolean {
  return !!tracks?.some(t => t.index === selected);
}

export function streamLabel(stream: AudioStream): string {
  const details = [stream.language, stream.title, stream.codec,
    stream.channels ? stream.channels + "ch" : null,
    stream.sample_rate_hz ? Math.round(stream.sample_rate_hz / 1000) + "kHz" : null,
  ].filter(Boolean).join(" · ");
  return "Audio " + (stream.index + 1) + (details ? " (" + details + ")" : "");
}
