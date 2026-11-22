import type { AudioStream } from "./api";

export interface ExplainedOption {
  value: string;
  label: string;
  description: string;
}

export const ASR_MODES: readonly ExplainedOption[] = [
  {value: "speaker", label: "Speaker-attributed (recommended)",
   description: "Preserves spoken words and fillers with punctuation and speaker tags. Best general choice for dialogue and subtitle timing; speaker IDs are reconciled later."},
  {value: "multispeaker_sot", label: "Overlapping speakers",
   description: "Prompts a speaker-tagged, punctuation-free transcript with attention to speaker turns and simultaneous speech. Useful for overlap-prone audio; not a guarantee of perfect separation."},
  {value: "verbatim", label: "Verbatim transcript",
   description: "Keeps fillers and repetitions, with punctuation, but does not request speaker tags. Use when literal words matter more than speaker attribution."},
  {value: "rich", label: "Rich transcription",
   description: "Requests special event and speaking-style tokens in addition to spoken words. Experimental for subtitles; tag support depends on the ASR model and response."},
];
export const SOURCE_REFINEMENT: readonly ExplainedOption[] = [
  {value: "off", label: "Off: keep source transcript",
   description: "No text-model source correction. Original ASR wording is used directly for translation."},
  {value: "conservative", label: "Conservative",
   description: "Correct obvious ASR slips, names and punctuation only when strongly supported by nearby dialogue."},
  {value: "balanced", label: "Balanced (recommended)",
   description: "Use context and optional glossary to fix likely mistranscriptions or homophones in the original spoken language without changing its meaning."},
  {value: "expressive", label: "Expressive",
   description: "Repair severely awkward or corrupted ASR wording more aggressively in the original language. Higher risk of overcorrection."},
];
export const TRANSLATION_REVIEW: readonly ExplainedOption[] = [
  {value: "off", label: "Off: raw machine translation",
   description: "Keep the dedicated translation model's English output as-is. Machine translation remains available for inspection."},
  {value: "conservative", label: "Conservative",
   description: "Correct clear translation errors, misrendered names and obvious inconsistencies against source dialogue."},
  {value: "balanced", label: "Balanced (recommended)",
   description: "Review the dedicated translation against original dialogue, neighboring speaker turns and glossary; fix supported semantic and idiomatic errors."},
  {value: "expressive", label: "Expressive",
   description: "Polish dialogue and emotional tone more strongly while preserving named entities, events and speaker intent."},
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
