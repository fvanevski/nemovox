import type { Job } from "./api";

/**
 * Final results must not reuse a mid-job cache entry when status polling
 * observes "complete" before the slower result polls.
 */
export function resultQueryKey(
  resource: "preview" | "artifacts",
  jobId: string | null,
  state: Job["state"] | undefined,
) {
  return [resource, jobId, state === "complete" ? "final" : "working"] as const;
}

export type MediaPreviewState = "loading" | "pending" | "ready" | "unavailable";

/** Do not load /media while a remote URL still has no downloaded source path. */
export function mediaPreviewState(job: Job | undefined): MediaPreviewState {
  if (!job) return "loading";
  if (job.request.source.kind !== "url" || job.source_path) return "ready";
  return job.state === "queued" || job.state === "running" ? "pending" : "unavailable";
}
