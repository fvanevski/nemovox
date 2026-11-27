# Connected translation qualification

This branch introduces development-only candidate code, not a new default
workflow. Central will decide production integration from the evidence. Do not
modify the normal CLI, Studio, renderer, source contracts, model choice or
review mechanisms in this campaign.

## Execute as Build, with ordinary local host authority

Fetch origin issue/25, verify fetched commit against the Central kickoff,
then make an isolated worktree at that exact commit. Preserve the owner's
checkout and existing jobs. Use a new private scratch root (umask 077;
mktemp -d with an issue-specific prefix); never delete an existing directory.
Run from the candidate root; use the owner's established orchestration
interpreter /workspace/nvidia-transcribe/.venv/bin/python if still authentic.
Verify imports resolve to the candidate, not the owner checkout. Respect AGENTS.md.
Do not use a retired V1 operational-schema gateway.

CPU:
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest tests/test_connected_translation_experiment.py -m "not gpu"
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -m "not gpu"
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q experiments

No frontend changes; no duplicate frontend build required at this experimental
boundary. Report failures honestly. Apply only safe formatter/whitespace fixes
if needed; do not alter behavior, tests, dependencies or acceptance criteria.
No remote source pushes or PR creation; return any exact mechanical diff to
Central. Correct host invocation/configuration binding, output paths and
window selection directly; no further permission required for ordinary
within-scope integration. For a substantive defect return the smallest
reproduction and actual traceback, continue independent cases where possible.

## One consolidated comparison campaign

Use retained accepted #24 MOSS French and Mandarin contracts, verified against
the private #24 receipts and model/media identity. Do not rerun MOSS merely
to recover already available evidence. Review the original French failure
provenance for U062–U063, U204/U206/U207, U217–U218, U100 and U193. Map each
to actual current event IDs, original wording, intended meaning, speaker and
interval using the old private job and accepted #24 source. Explicitly label
changed/resolved/missing/ambiguous source; never force equivalence. The private
raw audio and source context may help adjudicate material meaning; model
self-approval is not reference truth. Label language comprehension and
speaker/audio limits. Select contiguous windows with actual preceding/following
dialogue, not a concatenation of nonadjacent rows. At most 40 events/window,
80 total events per invocation, and disjoint windows. Use multiple windows
in one invocation to avoid repeated Riva loads. Use independent Mandarin
dialogue including clean sentences, genuine repeats, entity/pronoun ambiguity,
overlap and untimed events if present; include independent French controls
as appropriate. Prefer two to four bounded windows per recording, expanding
only enough to cover the named cases/controls; no full episode rerun now.

First run --plan-only for each selected set, inspect plan.json and group
decisions. Adapt window edges and add a bounded case if no multi-event
candidate exists. Never change grouping heuristics or canonical source to
manufacture candidate examples. If a named fragment now forms a single
MOSS event, report that as upstream resolution, then test remaining real
connected events separately. Do not assume grouping is beneficial.

Example (substitute real IDs and paths, using actual approved job resource
and glossary selections, not example defaults):

  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m experiments.connected_translation --contract "$CONTRACT" --window "$FIRST:$LAST" --window "$FIRST2:$LAST2" --glossary "$GLOSSARY" --model-resources "$RESOURCE_CONFIG" --output "$PRIVATE_ROOT/fr-plan" --plan-only

Repeat without --plan-only using a new output path (fr-live). Supply
--lifecycle-config only if the established host uses a nondefault owned
lifecycle config. Match accepted profile/style and glossary. Omit glossary
only when the accepted job did not use one. The resource resolver and
managed lifecycle use current native Riva and NInfer. Authenticate worker
interpreter, installed assets and model binding before running; explicit
cuda:0 only. Inspect nvidia-smi, owned unit identity and current capacity.
Do not run alongside an incompatible active local inference workload. Use a
capable independently hosted/CPU or remote-model agent, or normal operator
post-exit execution if lease conflicts require it. Never evade leases,
stop unrelated services, download models or use silent CPU fallback.
Check actual restored health and no orphan worker at completion.

The experiment executes:
- normal per-occurrence Riva with the production literal examples;
- strict and expanded connected-unit partitions (bounded gap, same
  speaker, native acquisition adjacency, no overlaps or row-bound evidence
  flattening; expanded allows full-stop crossing as a falsifiable alternative);
- paired per-constituent Riva with the SAME ordered examples as joined MT,
  deduplicated identical requests, one model load per invocation;
- existing NInfer review at each granularity, with primary and reviewed
  strings, actual private requests/decisions, and real SRT/ASS rendering.
  Review context is deliberately the same surrounding dialogue window,
  not an identical prompt/token sequence; unit granularity changes context.
  Grouped envelope is a display/review adapter only, not acoustic truth.
  Do not call the experimental projection a canonical source contract.

Compare primary joined vs matched separate on the same constituent IDs.
Compare normal primary vs normal reviewed, then grouped primary vs grouped
reviewed; do not attribute all gains to grouping. Report actual differences
in example selection, request counts, tokens, wall time, VRAM, review
failures/deferred decisions, omitted/additional meaning, polarity, direction,
pronouns, entity references, speaker/chronology, repeated content and
unrenderable/conflict/CPS results. Do not use edit count or CPS as a
semantic quality score. Report unchanged and adverse results. Review
requests are sensitive: private only. Evaluate potential entity consistency
using supported source/explicit glossary evidence; no guessed identity or
show-specific rules. Include a criterion-level case table with current IDs,
old correspondence if supported, reference meaning/evidence/uncertainty,
normal primary/reviewed, both candidate primary/reviewed, paired separate,
and correction/regression/residual assessment. Do not invent a gold
translation; distinguish language review, source-context inference and
unverified acoustic truth. Record whether useful grouping survives controls
and whether current review actually improves it. If unit-level review
retains harmful decisions, identify exact proposal/comparison reason and
request; do not add a model stage locally. If group admission excludes
material uncertain rows, document the exact rejected reason and remaining
architectural question rather than discard evidence.

## Evidence and stop

Return one private tar.gz containing complete bounded plan/request/result
JSON, review traces, case mapping/meaning assessment, source/asset/resource
identity (no secrets), CPU test logs, resource before/after, timings, cue
outputs and comparison, command/exit ledger, manifest and SHA256SUMS.
Include original #24 artifact hashes and bounded necessary evidence rather
than full media, full episode transcripts, weights, credential files, venvs
or private unrelated logs. Keep archive outside the repo and upload to this
ChatGPT conversation; report absolute local path, SHA256, size, executed full
commit and current remote branch ref. Do not push private evidence to
GitHub. If upload is unavailable, retain locally and return sanitized summary
plus precise missing delivery condition. No new PR, merge, closure, source
promotion or production deployment. Stop after the comparative campaign,
not after every case. Central will adjudicate the selected mechanism and
implement the production successor on the same branch.

## Tool routing and handback schema

Use Serena for bounded symbol/interface navigation if needed; Probe only
for supplemental extraction, RTK for routine noisy output only when it
preserves evidence, native Git/host commands for exact identity, failure
logs and resource evidence. OpenViking is historical context only, never
the authority for current source or host state; do not store private job
evidence there. Use the actual harness permissions, not historical V1
admission commands.

For a newly fetched candidate, run the following steps separately from the
owner repository before creating the isolated worktree:
  git fetch origin refs/heads/issue/25
  git rev-parse FETCH_HEAD
  git cat-file -t "$TARGET_SHA"
  git worktree add --detach "$PRIVATE_ROOT/worktree" "$TARGET_SHA"
Use TARGET_SHA from the exact Central kickoff; require FETCH_HEAD equality
and object type commit before worktree creation. Read the actual worktree
identity with native git and execute commands with that workdir. If the live
guard rejects a shape, honor the documented correction or report blocked.
No overwrite/reset/clean of any existing workspace.

Riva and text review run sequentially. Review uses one managed client per
invocation; when two arms have exactly identical primary rows within the
same case, an explicitly identified review receipt is reused rather than
repeating the same call. Count incremental cost only once. If the agent's
own model/lease prevents real inference, finish CPU and plan work and
prepare the exact post-exit commands with real private paths, authenticated
resources and selected windows, for operator execution using the same
candidate. Do not launch GPU inference in a conflicting active harness or
declare GPU acceptance from CPU results.

Return the following canonical friction ledger even when empty:

COMMAND_SHAPE_FRICTION:
  encountered=no
  events: []
  omitted_sensitive_values=no

For each material event set encountered=yes and record:
  - scenario=<bounded scenario key>
    requested_shape=<sanitized submitted invocation>
    effective_shape=<actual shape|same-as-requested|unknown>
    rewrite_provenance=<none|rtk_opencode_hook|shell_or_alias|other_harness_hook|unknown>
    event_kind=<pre_execution_rejection|execution_failure|successful_recovery|guard_annotation|harness_rewrite|runtime_misbinding|interface_discovery|other>
    execution_effect=<not_executed|executed_failed|executed_succeeded|recovered_after_retry|unknown>
    correction_or_annotation=<bounded actual diagnostic>
    replacement_shape=<corrected shape|none>
    recipe_reference=<GHDEV-10 recipe key|none>
    resource_reference=<plugin/repository resource|none>
    contract_conflict=<yes|no|uncertain>
    durable_gap_candidate=<yes|no|uncertain>
Set omitted_sensitive_values truthfully. Separately return
COMMANDS_MATCH_HANDOFF=yes/no. Preserve failures and successful recoveries,
even when the substantive result is positive. Never bypass a live guard.
