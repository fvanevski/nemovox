import { describe, expect, it } from "vitest";
import { emptySource, sourceForSubmission, uploadResponseIsCurrent } from "./source-selection";

describe("Studio media-source submission invariant", () => {
  it("accepts only sources belonging to the active method", () => {
    const host = {kind: "path" as const, value: "/media/clip.webm"};
    const contract = {kind: "contract" as const, value: "/media/recording.contract.json"};
    const uploaded = {kind: "upload" as const, value: "/uploads/clip.webm"};
    const remote = {kind: "url" as const, value: "https://example.org/video"};
    expect(sourceForSubmission("browse", host)).toEqual(host);
    expect(sourceForSubmission("browse", contract)).toEqual(contract);
    expect(sourceForSubmission("upload", host)).toBeNull();
    expect(sourceForSubmission("url", uploaded)).toBeNull();
    expect(sourceForSubmission("browse", remote)).toBeNull();
    expect(sourceForSubmission("upload", uploaded)).toEqual(uploaded);
    expect(sourceForSubmission("url", remote)).toEqual(remote);
    expect(sourceForSubmission("url", emptySource("url"))).toBeNull();
    expect(sourceForSubmission("browse", {kind: "path", value: "  "})).toBeNull();
  });

  it("does not admit an upload response after a mode change or newer upload", () => {
    expect(uploadResponseIsCurrent("upload", 3, 3)).toBe(true);
    expect(uploadResponseIsCurrent("browse", 3, 3)).toBe(false);
    expect(uploadResponseIsCurrent("url", 3, 3)).toBe(false);
    expect(uploadResponseIsCurrent("upload", 3, 4)).toBe(false);
  });
});
