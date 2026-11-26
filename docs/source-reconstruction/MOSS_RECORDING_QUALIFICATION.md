# Native MOSS recording qualification (#24)

## Authority and scope

Operator authorized event-centered MOSS integration, with coherent
meaning-bearing passages the normal unit. Fragmented crops, forced alignment,
independent ASR and speaker diagnostics are optional investigation tools;
they are not requirements or routine recovery stages. The preceding
comparative evaluation was accepted with corrected findings. Do not implement
another Nemotron ordered-anchor or general recovery selector.

Central has authored the production adapter, isolated worker, explicit
resource admission, actual native event validation, content-bound resume,
CLI and subtitle pipeline routing, native event protections, Studio worker
status and CPU regressions. The existing deployed default remains Nemotron
until Central adjudicates current episode-scale evidence. Both paths coexist
only for candidate qualification and the current installed default; do not
preserve superseded code indefinitely. This is one integrated implementation
slice, not a further model survey or an unconditional model promotion.
No #25 linguistic-unit redesign, automatic acoustic challenges, serving
infrastructure, additional models or new production recovery architecture.
Use the existing public process-media and generate-subtitles routes. No
production deployment or PR creation yet.

Baseline: prior comparative instruction 6aaa4fc826f77702fc2e0aee2e67295e54f7127e
(tree b525b16b43d023e4bc174fb1764a5042f13ee074), production baseline
4e813da48b2193d13190e0d200f75dbd1b91b35d. Exact successor commit/tree is
given in operator kickoff. Verify origin, expected HEAD and complete delta.
Main remains 92d9f1bde9eae00e467bfd0d8e804599c60ad87b. Read AGENTS.md
and model-resource LIFECYCLE.md. Candidate is not yet locally tested.
Do not make local changes merely to force old fixtures green.

## 1. Admission and CPU validation

Fresh private root beneath owner outputs/handbacks, mode 0700, isolated
detached worktree, owner checkout/status unchanged (including pre-existing
untracked files). Verify all mutable identities independently. Authenticate:
- /workspace/nvidia-transcribe/outputs/handbacks/model-comparison-6aaa4fc8.KjoDHIFs/nemovox-issue24-model-comparison-6aaa4fc8-review.tar.gz
  SHA256 7ba176f57d020f3adf108233012c1386384a8b4112c27fc75be249dcace61a88,
  159503 bytes, 48 regular files, 47 checksums. Inspect actual receipts,
  not the incorrect summary values. Reference all prior baseline evidence
  at original identities. No historical replay campaign.
- Existing MOSS local snapshot at comparative root models/MOSS-Transcribe-Diarize,
  HF revision 10304d926740cf5a5cee05a268086e3265197910;
  upstream package revision 79c675f0670cc2f824bcbcdd98263704f40f1f7e.
  Find actual previously used isolated interpreter from campaign environment
  evidence and readback; don't guess directory name. Compare actual
  package freeze (torch 2.11.0+cu128, transformers 5.24.1,
  moss-transcribe-diarize 0.1.0) and installed source against archived
  identity. Hash all model assets including actual weights and custom code,
  compare to resolved snapshot metadata and record actual revision and
  local hashes. Prior archive sizes/revision are not complete hashes.
  Inspect admitted custom code; no new downloads or version upgrades.
  Use existing isolated environment unchanged; if required, narrowly recreate
  that exact environment from approved cached assets, no owner venv changes.
- Copy effective existing model-resource JSON into private root (mode 0600;
  preserve credential indirection). Add a NEW named moss_local resource:
  {"adapter":"moss_events","model":"OpenMOSS-Team/MOSS-Transcribe-Diarize",
   "checkpoint_path":"<actual verified local snapshot>",
   "worker_python":"<actual isolated worker python>","device":"cuda:0"}.
  Set roles.asr to moss_local in the private copy. Do not overwrite asr_local
  or mutate deployed config. Use this private registry via --model-resources
  or NEMOVOX_MODEL_RESOURCES only for candidate subprocesses.

From candidate root:
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src/nvidia_transcribe
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q tests/test_moss_recording.py tests/test_model_resources.py tests/test_resource_integration.py tests/test_resource_lifecycle.py tests/test_recording_and_normalization.py -m "not gpu"
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m "not gpu"
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv-studio/bin/python -m pytest -q tests/test_studio_api.py tests/test_studio_export_api.py tests/test_studio_exports.py
Prove imports for nvidia_transcribe, nvidia_transcribe.moss_recording,
nvidia_transcribe.subtitle_pipeline and scripts resolve candidate, not main.
Do not repeat unrelated frontend build unless actual frontend files change;
web/README.md does not change executable frontend. Record previous equivalent
frontend identity. Check diff --check. Offline fixture tests should exercise
actual MOSS compact stream (not line-oriented), overlap, repeated wording,
valid short dialogue, raw span and interval bounds, malformed tail,
generation cap, device/identity, no lexical unit fabrication, resume and
both direct/retained routing. Parse actual four archived MOSS generations
using the new parser (duration from actual corpus, no inference or exact
output preservation gate). Compare parsed rows/text/intervals to archive
and report intentional 30.04 -> 30.0 end normalization; reject dropped
speech. No new test fixtures made from private source text in Git.

Local Build may fix demonstrated in-scope mechanical integration defects
(API argument shape, imports, pinned runtime differences, metadata keys,
ordinary exception handling and tests that accurately encode current
invariants). No architectural changes, fallback algorithms, fabricated
CTC records, weakening tests, boundary/admission relaxation or unrelated
cleanup. If a source correction is necessary, make the smallest correct
change, add a focused regression, rerun applicable gates. Maintain one
branch; do not commit/push private files. Local agent may commit only these
bounded corrective changes on issue/24 and push ordinary fast-forward after
fresh remote exact-head check; no force, reset, rebase, PR or main mutation.
If head drift, ambiguity or nonmechanical design needed, return diff and
evidence without push. A locally changed head requires exact final full
test and live identity; report final SHA/tree and full delta. Do not push
solely to publish evidence.

## 2. GPU lifecycle and full-recording qualification

Remote-backed capable harness, sequential GPU use. Inspect nvidia-smi and
owner job/Studio/service/lease status; do not stop unrelated services.
Worker uses normal ModelLifecycle.worker capacity and cooperative lock;
do not hold outer lock across managed commands or replace lifecycle.
Monitor actual GPU memory, wall and load, allocated/reserved peaks, actual
resource identity; no CUDA->CPU fallback or new inference server. Prefer
one complete recording request as tested architectural hypothesis, not
a claim of guaranteed long-form quality. Source language explicit. The
candidate uses full recording irrespective of legacy chunk parameters;
length-scaled max generation tokens, no acceptance at cap. Record actual
input prompt and generated token count, output final locality and any
early stopping/truncation suspicion. Model labels are anonymous and scoped
to the request. Never claim global character identity.

Select one full representative Mandarin episode (20-30min preferred):
prefer authenticated original episode containing D1 if accessible and
actual parent is full episode; inspect prior qualification media evidence
before assuming zh-scene.mkv is full length. Otherwise select an accessible
independent complete Mandarin episode, preferably known A Will Eternal
EP166 or another 20-30min dialogue-rich episode. If only the authenticated
H1 parent Heaven Official's Blessing S01E01 (2103.98s) is available, use
that, explicitly record duration deviation. Freeze selection, full SHA,
stream, actual source language, duration and useful preselected inspection
windows before seeing new outputs. No unrelated media search outside
authorized library. If reference source subtitles exist, authenticate;
English subs not exact Mandarin reference. No fabricated speech truth.

French full control: /opt/mcog/Les mystérieuses cités d'or Saison 4 Episode 1 [YlVGUDNbyXk].webm,
SHA256 0e77ce3c809e498b97ffe48b53575c29e10d8b08a039b78e2150cd0c845c81b0,
1379.901s, audio ordinal 0 actual French. If unavailable, report missing
input and use exact previously decoded F1a/F1b from comparative root as
bounded control; do not call it full French qualification.

First use archived short native output on CPU; optionally one exact D1
smoke via public process-media if needed to catch integration/runtime
failure before long run. No rerun all comparative arms. For each full
episode use the actual public path from detached root (PRIVATE_ROOT,
MEDIA, LANG, JOB and RESOURCE_CONFIG resolved to verified local values):

  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m nvidia_transcribe.cli process-media "$MEDIA" --output-dir "$JOB/asr" --source-language "$LANG" --mode speaker --audio-stream 0 --speaker-embeddings none --alignment-fallback off --model-resources "$RESOURCE_CONFIG"

Expected contract native_dialogue schema, actual raw output and manifest,
finite positive intervals or explicitly unlocalized rows, no fake CTC or
word times, same-request speaker labels, raw spans and hash binding. Record
native_dialogue generation hash, actual waveform hash, decoded duration,
token accounting, native output shape and warnings. Verify actual source
text matches each native span; event IDs unique; distinct repeated and
overlapping events are preserved. Do not equate empty/early terminating
generation with silence. Check audio near beginning/middle/tail and known
important windows, timeline/interval distribution, long gaps, phrase
repetition at competing localities, recurring speakers and anomalous label
fragmentation. Compare French echo and exposition intervals from retained
archive; compare D1/H1 if included in full episode, otherwise use retained
short evidence and preselected independent windows. Actual quality,
not arbitrary row-count or timing coverage targets. Compare timing
granularities separately; no ground-truth CER/DER unless reference exists.

If full-recording input fails materially (OOM, cap, parsing, missing long
passages, speaker collapse), diagnose actual failure. Allowed one bounded
alternate geometry study, NOT production fallback: if full recording
resource/cap fails, use two consecutive approximate half-episode
windows with actual exact decoded offsets, same adapter and one request
per window; if localized coverage/recognition failure, select one coherent
larger scene window containing defect + surrounding dialogue. Record
separate speaker scopes and all text, no stitched canonical output,
forced mapping, per-word crops or mandatory differential chunking.
Do not retry endless geometries. If full episode unusable, retain failure
and alternative evidence; stop its downstream normal qualification
rather than pretend partial is accepted. Continue independent French
and applicable CPU/Studio checks. Central will decide any geometry policy.

## 3. Integrated translation and rendering

For each usable fresh recording, exercise same ordinary continuation as
Studio, current Riva and balanced NInfer English review, no #25 redesign:

  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m nvidia_transcribe.cli generate-subtitles "$MEDIA" --recording-contract "$JOB/asr/recording.contract.json" --output-dir "$JOB/output" --source-language "$LANG" --target-language en --source-quality auto --refinement off --translator riva --translation-review balanced --style natural --formats srt,ass --ass-layout standard --ass-max-lanes 3 --model-resources "$RESOURCE_CONFIG"

No legacy lexical source-review invocations; expected explicit not_run
reason, not a source review success. Preserve raw source, native times and
speakers through refined/translated contracts. Verify Riva MT and bounded
NInfer Messages review actual receipt, deferred/errors, no silent fallback,
source row IDs, genuine repeats, short turns, overlap, usable SRT/ASS.
Inspect representative actual English and meaning errors, particularly
relative-month phrase where relevant. Do not repair #25 semantic review
by relaxing guards. Translation failure may be isolated from source
assessment; record scope and one normal same-job continuation if appropriate.
Render failure not disguised as successful subtitles. Do not publish to
user media library.

Same-job process-media re-invocation after successful native acquisition:
require exact contract/manifest/raw generation bytes, no new model
acquisition, no GPU worker. Same-job generate-subtitles continuation
require no unnecessary ASR and unchanged output where review/MT complete.
CPU-only offline --re-render using identical output directory and options;
compare SRT/ASS digests and translated contract identity. Record cache
hits/actual inference counts rather than merely elapsed time. Offline
rerender must not load MOSS.

Run real isolated Studio backend smoke with private job root and private
registry, using existing application factory/config, never deploy/stop
active owner Studio. Prefer one retained valid contract import and
one short source-media transcription job (reuse existing D1 smoke or
French bounded case; if no runtime need, mocked route regression plus
actual short job). Verify /api/resource-status startable_local_worker
without false endpoint claims, job events, exports and actual native
recording path; ordinary subtitle continuation route already qualified
through CLI. Do not add new frontend controls or change default UI.
Resource-state, cancel/process group and status checks required if affected.
If startup would reclaim active unrelated/owner services, use read-only
test harness and report blocked live smoke, no service mutations.

## 4. Handback and Central decision

One private mode-0600 tar.gz, no raw audio/media, model assets, venv,
credentials, private registry contents, unrelated files or huge logs.
Include admission.json, source delta/test receipts, resource identity
and complete model asset hash inventory, environment/package identity,
frozen corpus.json, commands/exit status/wall/VRAM receipts, per-case
native-dialogue.raw.json, manifest.json, recording.contract.json, transcript,
refined/translated contracts, SRT/ASS and metadata; actual bounded
comparison/repeat/speaker/coverage/quality assessment and adverse cases.
Use unique relative paths, include all necessary current result evidence,
sanitized bounded logs, exact model and source identities.
Include explicit continuation/rerender evidence, Studio tests/smoke,
restoration.json, COMMAND_SHAPE_FRICTION.json per GHDEV-10 (zero if none),
HANDBACK.md, archive manifest.json and SHA256SUMS. Report required vs
included/missing artifacts, raw output digests and reference/proxy
authority, archive safety and privacy checks. Archive manifest lists
path/size/hash/criterion/source+runtime identity/limits, no circular hashes;
SHA256SUMS covers manifest and all payload. Verify safe unique regular
members and exact checksum inventory, archive readback SHA256/bytes.
No private evidence upload to GitHub. Return archive path, digest, bytes
and direct operator attachment action.

Acceptance for this slice: current implementation passes applicable
deterministic tests; actual model acquisition and event contract on
episode-scale Mandarin and French control; useful observed dialogue,
absence of catastrophic omitted/displaced passages or speaker collapse
in inspected controls; valid native event timing and explicit unresolved
material errors; normal continuation/render and no unsupported recovery.
Full-episode model-generated transcript and labels are not independent
ground truth. Report source candidate PASS/FAIL/BLOCKED separately from
downstream translation/render and Studio. Retain original output, do not
hide deficiencies. A source-quality pass does not imply #24 closure or
#25 acceptance. Central will determine primary-model promotion, selective
investigation utility, and clean-forward removal of obsolete CTC recovery.
No PR, main mutation, deployment or issue closure until Central reviews.
