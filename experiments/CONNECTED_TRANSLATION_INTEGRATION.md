# Connected translation: normal workflow qualification

This is the next implementation qualification for issue #25, not a new
exploration campaign. Central authored the production candidate and successor
tests. The rejected strict/expanded experimental arms are diagnostic only.
Do not promote or replay those arms. The operator kickoff supplies the exact
TARGET_HEAD_SHA and tree identity; this document intentionally cannot contain
the SHA of its own commit.

## Scope and responsibilities

Primary local agent: Build, fresh session. This is host integration and
qualification with bounded mechanical repair, not delegated semantic design.
Use the operator's existing Nemovox orchestrator/Studio interpreters and
configured owned services. Preserve the owner main checkout, installed models,
original media, previous jobs and unrelated services. No deployment, workflow
promotion, new PR, merge, issue closure, new model, #9 scheduler redesign or
other service changes.

Central decision: strict geometry nominates a connected Riva alternative;
it does not assert continuity. Every original occurrence gets independent MT.
Native review sees source observations with original boundaries and the
alternative, then compares the complete allocated proposal to the independent
primary allocation. One atomic verdict governs each connected block; separate
units cannot inherit unrelated rejection. Retain canonical IDs and actual
intervals; output allocations remain on original occurrences. No additional
default review or condensation pass. Existing explicit row-bound uncertainty
and polarity checks still apply. The current candidate requires nonempty
English for each occurrence; indivisible allocations and better display
segmentation remain a limitation to evaluate, not a license to invent timing.
Entity consistency uses source-bound context and operator bilingual examples;
there is no unqualified persistent episode entity memory.

Local authority: resolve resource-file binding, isolated test dependencies,
path/command friction, imports/formatting and truly mechanical defects without
changing semantic behavior, assertion meaning, lifecycle/security gates, source
contracts or model policy. Do not edit a semantic algorithm or invent a
replacement design. If a real mechanical repair is needed, retain the complete
diff and a local commit, do not push. Test the actual resulting tree and return
its SHA separately. Otherwise run the admitted exact candidate without edits.
If deterministic invariants fail, stop expensive dependent inference, preserve
a minimal reproducer and continue independent checks that remain useful.

## Admission and CPU phase

Use Bash for the following shell commands. Start with owner status and origin
identity, ensure origin is the intended repository, and confirm no unexpected
remote movement. Fetch the newly staged selector before proving the object.

    OWNER=/workspace/nvidia-transcribe
    git -C "$OWNER" status --short --branch
    git -C "$OWNER" remote get-url origin
    git -C "$OWNER" fetch --no-tags origin refs/heads/issue/25
    test "$(git -C "$OWNER" rev-parse --verify FETCH_HEAD)" = "$TARGET_HEAD_SHA"
    test "$(git -C "$OWNER" cat-file -t "$TARGET_HEAD_SHA")" = commit
    git -C "$OWNER" rev-parse "$TARGET_HEAD_SHA^{tree}"
    umask 077
    PRIVATE_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/nemovox-connected.XXXXXXXX")
    git -C "$OWNER" worktree add --detach "$PRIVATE_ROOT/worktree" "$TARGET_HEAD_SHA"
    cd "$PRIVATE_ROOT/worktree"
    PY="$OWNER/.venv/bin/python"
    STUDIO_PY="$OWNER/.venv-studio/bin/python"
    PYTHONPATH=src:. "$PY" -c 'import nvidia_transcribe.translation as t; import nvidia_transcribe.connected_units as u; print(t.__file__); print(u.__file__)'
    PYTHONPATH=src:. "$PY" -m pytest tests/test_connected_units.py -m 'not gpu'
    PYTHONPATH=src:. "$PY" -m pytest -m 'not gpu'
    PYTHONPATH=src:. "$PY" -m compileall -q src/nvidia_transcribe
    PYTHONPATH=src:. "$STUDIO_PY" -m pytest tests/test_studio_api.py tests/test_studio_export_api.py tests/test_studio_exports.py -m 'not gpu'

Record exact commands, exit codes, raw stdout/stderr and source import identity
for both interpreters. Use current repo-native Studio/frontend scripts in the
isolated worktree (inspect web/package.json and lockfile; lockfile-pinned
npm ci if necessary, npm test, npm run build). Do not modify owner dependencies,
tracked lockfiles or main source. No tests are to be skipped or relaxed to get
green. Old experimental tests remain diagnostic until accepted final cleanup.

## GPU phase — actual normal pipeline, not a synthetic adapter

Authenticate the retained source contracts against the private evidence archive
from the preceding qualification (SHA-256
0e962c2701800116b220a288017b6fac172371d5b2bf0640128604b68d53254e).
Use the accepted #24 promotion output root under the owner's outputs/handbacks
directory. Retained input identities:

- runs/fr_s04e01/recording.contract.json: SHA-256
  affe54d99cbed6c4e23870646a96eca19d723790ac7d011684867192cf38b9c3,
  257 events.
- runs/zh_ep166/recording.contract.json: SHA-256
  ac577bdebf534a5e89b33b845616fc20b9418e6798fa97a6b43a7a5ebd19de56,
  329 events.

Use actual resolved input paths, not paths invented from these labels. Validate
the actual MOSS contract and model-resource role/adapter identity. Select
configured Riva cuda:0 and the owned NInfer review resource as in prior evidence;
do not stop external services, load a different checkpoint, download models or
fall back to CPU. Obtain resource/service and VRAM before/after each live job.
An agent harness occupying GPU capacity must be moved off that GPU by the
operator, not by automatic service eviction.

Run each retained episode once through the normal candidate CLI with the same
balanced/natural profile and no glossary as the prior comparison. Use a fresh
private output directory per job. The source is retained to avoid redundant ASR;
the translation and contextual review are fresh. Example (bind variables from
host facts, not literals containing angle-bracket placeholders):

    PYTHONPATH=src:. "$PY" -m nvidia_transcribe.cli generate-subtitles \
      --recording-contract "$FR_CONTRACT" --output-dir "$PRIVATE_ROOT/fr" \
      --model-resources "$RESOURCE_CONFIG" --translator riva \
      --translation-review balanced --style natural --source-quality auto --fresh

Repeat once for Mandarin with its verified contract. No --condense-high-cps,
no sidecar directory, no active Studio job mutation. The accepted #24 full
translated contract, when available and authenticated against identical source,
resource/settings, supplies a full-episode comparator; otherwise compare
retained normal/strict/expanded bounded case evidence. Different review window
geometry is not a controlled causal isolation; record it as a limitation.
No need to rerun old arms or historical Nemotron transcription.

Collect per-job summary, input/source inventories and hashes, primary separate
MT, connected alternatives with request identities/examples/tokens, actual
proposal and comparison allocations, exact accepted receipts, retained English,
cue lineage and rendering metrics. Record Riva plus text-review time, token
receipts where present, and actual call/retry/defer/error counts. Candidate
Riva token totals are separate native_input_tokens plus connected_input_tokens.
Correct prior Mandarin experiment total: 3236, not 2707. Do not compare full
episode cost directly to 80-row arms without normalizing/reporting scope.

## Semantic acceptance cases

Inspect actual complete surrounding dialogue, not only one English string.
Use audio/context when available and necessary; report unverified acoustic
interpretation rather than assert model approval proves truth.

FR:
- 39–41 parallel infinitives, complete advice/translation/riddle meaning;
- 115–116 relative clause, no invented instruction;
- 236–237 causal connector; 71 westward direction, 125 name/referent,
  227 pronoun/action, 253 pebbles/game clause must remain accurate.
ZH:
- 82–83 recover the role-taking clause, not releasing the person;
- 87–103 verify independent review decisions are not reverted merely by
  proximity, inspect consistent supported entity terms and roles;
- 173–174 no vocative-to-possessive fusion;
- 191–195 preserve all five genuine repetitions/speakers;
- 196–198 preserve interrupted dialogue; upstream speaker ambiguity is not
  evidence for joining turns;
- 203–204 complete clause and supported river terminology;
- 260–261 and 273–274 parallel constructions;
- 265–266 preserve combat subject/object relationship;
- 275–277 preserve explicit form of address and idiom.

Report exact occurrence IDs, actual source and timing, separate/joint MT,
proposed and selected English allocation, decision receipt or reason for
no candidate, expected meaning, outcome (improved / equivalent / worsened /
unresolved), and any material new errors. Inspect independent previously
unselected examples of both languages, including complete sentences,
ambiguous syntax, repeated dialogue, speaker changes, overlap, untimed
events if available, and contextual entity mentions. If source inputs lack
an edge condition, do not fabricate a live case. Use deterministic controls
and obtain a genuinely independent short native recording with relevant
conditions when practical. No required correction count or percentage.
A sound algorithm may reject a bad joint alternative; low acceptance or
edit count alone is not failure.

Compare complete ordered occurrence IDs, source text, speaker labels,
source intervals, review decision ownership and final cue lineage against
input. Verify no hidden duplicate/loss or fabricated timing. Untimed dialogue
must remain explicit; renderer overflow/conflicts must be reported. Current
display subdivisions are allocation intervals, not new acoustic observations.
No broad layout optimization.

## Current-version continuation, optional paths, and final integrated run

After a successful retained job, rerun one job without --fresh into its own
existing private output directory with otherwise identical configuration;
capture cache hits, unchanged selected unit/English/source identity, and
absence of unnecessary fresh inference where checkpoints support it. Then
perform the normal --re-render command against the retained translated
contract; hash it before/after, compare SRT/ASS and cue lineage, verify no
model acquisition. Confirm review-off, same-language and explicit condensation
and override behavior through deterministic tests; if live optional path is
needed, use a short isolated input only. No separate long campaign.

If deterministic gates and both retained live cases show material improvements
without new material semantic or mapping regressions, proceed in the same
handoff to one fresh complete episode through normal CLI or isolated Studio
job and a short independent multilingual control (other language if possible).
Choose suitable available local media; never download copyrighted media or
change library files. Fresh ASR requires the authenticated local MOSS resource
and explicit spoken language; normal workflow settings, private outputs, no
sidecars. For Studio use the isolated candidate environment rather than live
deployment. If there is no suitable input, service admission is blocked, or
a material regression remains, record precisely what is blocked and return
the completed evidence. Do not keep rerunning until a favorable result.
The new implementation and both full retained translations are a coherent
qualification unit; fresh acceptance is conditional to prevent wasting GPU
time on a rejected candidate.

## Handback

Keep one private evidence archive outside the repository, mode 0600.
Include SHA-256 manifest/checksums, original and final target/tree/owner status,
complete bounded diff if any, test and build logs, import identity, resource
fingerprints and service/process/VRAM receipts, run commands/exit codes, input
and output contract hashes and inventories, case table and bounded private
source/translation/receipt excerpts, cue lineage/coverage and readability,
cache/rerender evidence, optional/fresh-run disposition, and a criterion-level
assessment with adverse and unresolved cases. Exclude secrets, credentials,
model weights, full-length media, large logs, virtual environments and unrelated
files. Do not publish private transcripts or the archive to GitHub; provide the
absolute path, bytes, SHA-256 and manifest and ask the operator to attach it
to Central ChatGPT. Do not claim upload if the harness cannot do it.

Return actual executed commit/tree, no remote updates unless explicitly
authorized by a subsequent Central decision, and whether owner checkout and
owned resources were restored. Report exact scope of any mechanical edits and
every command-shape deviation. Use the canonical GHDEV-10 friction ledger:

    COMMAND_SHAPE_FRICTION:
      encountered=no
      events: []
      omitted_sensitive_values=no

For material events record scenario, requested_shape, effective_shape,
rewrite_provenance, event_kind, execution_effect, correction_or_annotation,
replacement_shape, recipe_reference, resource_reference, contract_conflict,
durable_gap_candidate. Separately return COMMANDS_MATCH_HANDOFF=yes/no.
If no evidence is available, mark UNVERIFIED, not PASS.

Tools: Serena for relevant symbols/references if necessary, Probe only
supplemental, RTK only for routine verbose summaries where evidence is
preserved, native git/pytest/systemctl/nvidia-smi and original error output for
exact authority and diagnostics. OpenViking only for bounded durable context;
no private logs/media/transcripts/secrets or mutable authority claims. The
historical operational-schema gateway is not assumed active.

Central owns semantic adjudication, any necessary architecture corrections,
retirement of superseded experiments, single PR creation, independent exact
head review and merge/closure after explicit authorization.
