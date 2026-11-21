import type { MediaKind } from "./api";

export type SourceMethod = "browse" | "upload" | "url";
export type SelectedSource = {kind: MediaKind; value: string};

export function emptySource(method: SourceMethod): SelectedSource {
  return {kind: method === "url" ? "url" : method === "upload" ? "upload" : "path", value: ""};
}

export function sourceForSubmission(method: SourceMethod, source: SelectedSource): SelectedSource | null {
  const matches = method === "browse" ? source.kind === "path" || source.kind === "contract"
    : method === "upload" ? source.kind === "upload" : source.kind === "url";
  return matches && source.value.trim().length > 0 ? source : null;
}

export function uploadResponseIsCurrent(method: SourceMethod, submittedEpoch: number, currentEpoch: number): boolean {
  return method === "upload" && submittedEpoch === currentEpoch;
}
