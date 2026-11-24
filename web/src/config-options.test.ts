import { describe, expect, it } from "vitest";
import { INITIAL_OPTIONS, ASR_MODES, TRANSLATION_REVIEW, LOCALIZATION_STYLES,
  hasAudioIndex, normalizeAudioIndex, streamLabel, isEnglishSourceToEnglish } from "./config-options";
import type { AudioStream } from "./api";

const tracks: AudioStream[] = [
  {index: 0, container_index: 1, codec: "aac", language: "fra", title: "French",
   channels: 2, sample_rate_hz: 48000},
  {index: 1, container_index: 3, codec: "eac3", language: "eng", title: "English",
   channels: 6, sample_rate_hz: 48000},
];

describe("Studio configuration guidance", () => {
  it("describes ASR and the single target-language review rather than source correction", () => {
    for (const options of [ASR_MODES, TRANSLATION_REVIEW, LOCALIZATION_STYLES]) {
      expect(options.every(x => x.label && x.description.length > 35)).toBe(true);
      expect(new Set(options.map(x => x.value)).size).toBe(options.length);
    }
    expect(ASR_MODES.map(x => x.value)).toEqual(["speaker", "multispeaker_sot", "verbatim", "rich"]);
    expect(TRANSLATION_REVIEW.some(x => x.value === "balanced")).toBe(true);
    expect(TRANSLATION_REVIEW[1].description).toContain("translation");
    expect(INITIAL_OPTIONS.refinement).toBe("off");
    expect(INITIAL_OPTIONS.translator).toBe("riva");
    expect(INITIAL_OPTIONS.translation_review).toBe("balanced");
    expect(INITIAL_OPTIONS.condense_high_cps).toBe(false);
    expect(INITIAL_OPTIONS.chunk_policy).toBe("fixed");
    expect(isEnglishSourceToEnglish("en", "en")).toBe(true);
    expect(isEnglishSourceToEnglish("en-US", "English")).toBe(true);
    expect(isEnglishSourceToEnglish("eng", "en")).toBe(true);
    expect(isEnglishSourceToEnglish("auto", "en")).toBe(false);
    expect(isEnglishSourceToEnglish("fr", "en")).toBe(false);
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
