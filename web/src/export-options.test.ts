import { describe, expect, it } from "vitest";
import { availableArtifacts, selectedArtifacts } from "./export-options";

describe("Studio task-specific export selection", () => {
  it("keeps transcription output focused on source files", () => {
    expect(availableArtifacts("transcription").map(x => x.id))
      .toContain("source_transcript");
    expect(selectedArtifacts("transcription", ["source_transcript", "srt", "ass", "final_english"]))
      .toEqual(["source_transcript"]);
  });
  it("allows a translation task to export text and SRT but not ASS", () => {
    expect(selectedArtifacts("translation", ["final_english", "srt", "ass"]))
      .toEqual(["final_english", "srt"]);
  });
  it("supports subtitle SRT and ASS with stable deduplication", () => {
    expect(selectedArtifacts("subtitles", ["srt", "ass", "srt"]))
      .toEqual(["srt", "ass"]);
  });
});
