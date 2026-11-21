import { describe, expect, it } from "vitest";
import { secondsLabel } from "./api";

describe("secondsLabel", () => {
  it("formats zero and fractional seconds", () => {
    expect(secondsLabel(0)).toBe("00:00.00");
    expect(secondsLabel(12.34)).toBe("00:12.34");
    expect(secondsLabel(3661.5)).toBe("01:01:01.50");
  });

  it("handles null, undefined, and NaN", () => {
    expect(secondsLabel(null)).toBe("—");
    expect(secondsLabel(undefined)).toBe("—");
    expect(secondsLabel(Number.NaN)).toBe("—");
  });
});
