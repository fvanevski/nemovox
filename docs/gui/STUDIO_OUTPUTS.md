# Studio output destinations, exports, progress, logs and history

This is a **local, single-user** Studio. The directory selected in the browser
is a **host filesystem directory**, not a browser download destination. The
server validates the destination independently of the frontend.

## Host configuration and default

Set NEMOVOX_EXPORT_ROOTS to an os.pathsep-separated list of trusted host output
directories before starting Studio (e.g. a Videos or Downloads subdirectory).
For Linux, the separator is a colon. Do not include the entire filesystem, a
secret directory, a mounted private service tree or an untrusted shared location.

If unset, Studio uses /workspace/nvidia-transcribe/outputs/exports when
started from the normal checkout. The exact default is the repository
outputs/exports directory. It may be created on the first export; browsing it
does not write to it.

The Source tab provides a folder field and an authorized directory picker.
Subfolders within an export root can be entered and are created when needed.
A symlink escaping an approved root is rejected.

## Full internal archive versus selected deliverables

Every job always retains its output under
NEMOVOX_STUDIO_JOBS/<job-id>/{asr,output,remote}, plus job.json, execution.log
and exports.json where applicable. Only selected final types are copied to
the chosen host output folder. Each job gets a collision-safe child directory:
<output-folder>/nemovox-<full-job-id>/. The export report records source,
destination, status, SHA-256 digest and individual error if any. Retrying an
identical copy is idempotent; a different preexisting file is never replaced.

Users can select source transcript, final English dialogue, SRT, ASS and
individual provenance/metadata JSON or text files appropriate to the task.
Internal work products continue to be generated even when not selected for
export. The translation task preserves its current legacy internal SRT
generation contract, but SRT export is optional.

With a URL input, "Also save downloaded original media" copies the downloaded
source into the chosen job output folder once the download completes, even if
subsequent ASR or translation fails. The internal original is always retained.
It is not automatically exported when the checkbox is off.

An export failure is reported separately from ASR, Riva translation, review
and rendering; success is never silently claimed for files that were not
written. Failed exports leave all internally generated files accessible.

## Stages and logs

Studio continues to run ONE generate-subtitles CLI subprocess. It consumes
versioned NEMOVOX_EVENT JSON progress records emitted at the actual translation,
English review, optional condensation and rendering boundaries, and stores a
stage history in job.json. Skipped same-language, disabled-review and offline
rerender paths are not recorded as successful inference.

The processing log is retained after all terminal job states. Studio shows
the live tail, supports copying visible content and bounded pagination for
copying the full log, and exposes the complete downloadable log. Job history
is paginated beyond the legacy recent-50 response; older job artifacts and
logs remain available while their job-specific directories are retained.

## Operational behavior

An omitted export plan retains internal artifacts only. Direct CLI commands
keep their output-directory and inference behavior. See the current
[Studio workflow](SUBTITLE_WORKFLOW.md) and [lifecycle guide](../model-resources/LIFECYCLE.md).
