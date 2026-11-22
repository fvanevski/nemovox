import { describe, expect, it } from "vitest";
import { ASR_MODES, SOURCE_REFINEMENT, TRANSLATION_REVIEW, LOCALIZATION_STYLES,
  hasAudioIndex, normalizeAudioIndex, streamLabel } from "./config-options";
import type { AudioStream } from "./api";

const tracks: AudioStream[] = [
  {index: 0, container_index: 1, codec: "aac", language: "fra", title: "French",
   channels: 2, sample_rate_hz: 48000},
  {index: 1, container_index: 3, codec: "eac3", language: "eng", title: "English",
   channels: 6, sample_rate_hz: 48000},
];

describe("Studio configuration guidance", () => {
  it("describes every option and separates source and target correction profiles", () => {
    for (const options of [ASR_MODES, SOURCE_REFINEMENT, TRANSLATION_REVIEW, LOCALIZATION_STYLES]) {
      expect(options.every(x => x.label && x.description.length > 35)).toBe(true);
      expect(new Set(options.map(x => x.value)).size).toBe(options.length);
    }
    expect(ASR_MODES.map(x => x.value)).toEqual(["speaker", "multispeaker_sot", "verbatim", "rich"]);
    expect(TRANSLATION_REVIEW.some(x => x.value === "balanced")).toBe(true);
    expect(SOURCE_REFINEMENT[1].description).toContain("ASR");
    expect(TRANSLATION_REVIEW[1].description).toContain("translation");
  });
  it("never invents audio tracks or preserves indices from a different file", () => {
    expect(tracks.map(x => x.index)).toEqual([0, 1]);
    expect(hasAudioIndex(tracks, 0)).toBe(true);
    expect(hasAudioIndex(tracks, 2)).toBe(false);
    expect(normalizeAudioIndex(tracks.slice(0, 1), 1)).toBe(0);
    expect(normalizeAudioIndex(tracks, 1)).toBe(1);
    expect(hasAudioIndex(undefined, 0)).toBe(false);
    expect(streamLabel(tracks[0])).toContain("French");
    expect(streamLabel(tracks[1])).toContain("eng");
  });
});
