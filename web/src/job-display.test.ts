import { QueryClient } from "@tanstack/react-query";
import { describe, expect, it } from "vitest";
import type { Job } from "./api";
import { mediaPreviewState, resultQueryKey } from "./job-display";

describe("final job results", () => {
  it("changes both result cache keys when a job completes, not when it merely runs", () => {
    for (const resource of ["preview", "artifacts"] as const) {
      expect(resultQueryKey(resource, "job1", "queued"))
        .toEqual(resultQueryKey(resource, "job1", "running"));
      expect(resultQueryKey(resource, "job1", "running"))
        .not.toEqual(resultQueryKey(resource, "job1", "complete"));
      expect(resultQueryKey(resource, "job1", "complete"))
        .not.toEqual(resultQueryKey(resource, "job2", "complete"));
    }
  });

  it("fetches a fresh result rather than reusing the intermediate response", async () => {
    const cache = new QueryClient({defaultOptions: {queries: {retry: false, staleTime: Infinity}}});
    for (const resource of ["preview", "artifacts"] as const) {
      let requests = 0;
      const read = () => ++requests;
      const pending = await cache.fetchQuery({
        queryKey: resultQueryKey(resource, "job", "running"), queryFn: read,
      });
      const final = await cache.fetchQuery({
        queryKey: resultQueryKey(resource, "job", "complete"), queryFn: read,
      });
      expect(pending).toBe(1);
      expect(final).toBe(2);
      expect(requests).toBe(2);
    }
    cache.clear();
  });
});

describe("remote media readiness", () => {
  const job = (kind: Job["request"]["source"]["kind"], state: Job["state"], path: string | null) =>
    ({state, source_path: path, request: {source: {kind, value: "synthetic"}}}) as Job;

  it("defers video loading until a remote download supplies its source path", () => {
    expect(mediaPreviewState(undefined)).toBe("loading");
    expect(mediaPreviewState(job("url", "queued", null))).toBe("pending");
    expect(mediaPreviewState(job("url", "running", null))).toBe("pending");
    expect(mediaPreviewState(job("url", "running", "/tmp/download.webm"))).toBe("ready");
    expect(mediaPreviewState(job("url", "complete", "/tmp/download.webm"))).toBe("ready");
  });

  it("does not conceal a failed remote download or block already-known local media", () => {
    expect(mediaPreviewState(job("url", "failed", null))).toBe("unavailable");
    expect(mediaPreviewState(job("url", "cancelled", null))).toBe("unavailable");
    expect(mediaPreviewState(job("url", "complete", null))).toBe("unavailable");
    expect(mediaPreviewState(job("path", "running", "/media/clip.webm"))).toBe("ready");
    expect(mediaPreviewState(job("contract", "complete", "/media/recording.contract.json"))).toBe("ready");
  });
});
