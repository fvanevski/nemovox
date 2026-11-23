# Issue #14 / PR #15: offline independent timing anchors and reviewed child-voice boundaries

LOCAL_AGENT_PRIMARY=build
REPO_ID=fvanevski/nemovox
ISSUE_NUMBER=14
PR_NUMBER=15
BRANCH=issue/14-longform-timing-boundaries
CENTRAL_SOURCE_SHA=7bb957ed829649d965445c0aee91bf062b698849
CENTRAL_SOURCE_TREE=f42962cfe3986b2af19bd2e23cfb8ea189a09370
BASE_MAIN_SHA=8f6c03cbc70e84bf1970ba364207a02c9fe22b44
CONTROLLING_DESIGN=staging/issue14/INDEPENDENT_TIMING_ANCHORS_DESIGN.md
PRIOR_PUBLIC_EVIDENCE=80617da496537cf5fdda55e7f7b26438069a9844:staging/issue14/evidence/20261123T174459Z/
PRIOR_PRIVATE_ARCHIVE=/workspace/nemovox-issue14-default-recovery-private/issue14-default-source-recovery-evidence-20261123T174459Z.tar.gz
PRIOR_PRIVATE_SHA256=1b98ad231c3a15e98da41c76a9056d636f49a07819e298cbd1b55c46663aac4f
NO_NEW_PR=yes
NO_MERGE=yes
NO_DEPLOY=yes
NO_ISSUE_CLOSE=yes
NO_NEW_FULL_EPISODE_ASR=yes
NO_MODEL_DOWNLOAD=yes
NO_PRODUCTION_SOURCE_CHANGE_BY_LOCAL_AGENT=yes

## Goal / why this is one bounded campaign

Prove or reject the new CENTRAL-authored *evidence-only* cross-ASR timing matcher on the original private French Nemotron source ledger plus previously captured faster-whisper large-v3 word outputs, then produce acoustic per-turn review evidence for W02b child speakers and a few sustained CTC drift cases. One consolidated host handback, not a general speech-model survey or another full French 23min rerun. Existing source, speaker, alignment, translation and Studio behavior must not change.

## 1. Fresh identity and CPU gates

Start one fresh OpenCode Build instance in a NEW isolated worktree of current authenticated PR branch. Fetch and verify exact live issue/PR/main/base refs and that CENTRAL_SOURCE_SHA is an ancestor of current PR HEAD. Check the latest commit's tree and verify no src/tests change after CENTRAL_SOURCE_SHA; AGENTS.md has new offline and Studio-import guidance. If newer src/tests changes exist, stop for Central rebind; no rebase/reset/clean/force affecting operator worktree. Verify private archive and its CHECKSUMS.sha256 (56/56), and the original public manifest. Keep /workspace/pyannote, its .venv, model snapshots, historic jobs and native host services untouched. Use accurate existing interpreter, previously /workspace/nvidia-transcribe/.venv/bin/python; do not create new venvs. Run actual CPU commands (using the candidate source):
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src/nvidia_transcribe
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu' tests/test_issue14_source_challenger_timing.py tests/test_issue14_timing_boundaries.py tests/test_issue14_source_unit_lineage.py tests/test_issue14_source_reconciliation.py
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu'
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv-studio/bin/python -m pytest -q -m 'not gpu'
  git diff --check
Before tests prove `python -c 'import nvidia_transcribe.source_challenger_timing as c; print(c.__file__)'` in EACH relevant venv resolves to isolated candidate, not main checkout. Do not use an unset PYTHONPATH for candidate Studio. Preserve exit statuses and complete failure log. No test skip/xfail or weakening. Unambiguous mechanical environment fixes allowed, source semantics not.

## 2. Exact offline source-timing replay, CPU-only

Create one NEW private review root, outside repository staging and historical output. Verify archive paths are regular safe files and member checksums, then extract only required source ledger, frozen windows, challenger results and bounded public records there. Run the documented private CLI from isolated candidate source:
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m nvidia_transcribe.source_challenger_timing --source-evidence <private-root>/default_fr_job/asr/source-evidence.v1.json --frozen-windows <private-root>/challenger/frozen_windows.json --challenger-results <private-root>/challenger/c2_results.json --model-key whisper_large_v3 --output <private-root>/fr_whisper_time_comparison.json
Use the output path as a previously non-existent private file (CLI creates mode 0600 exclusively). Also run turbo with a NEW distinct output path as negative/hallucination/fast-ASR comparison. Do NOT ask Whisper, WhisperX or pyannote to run again: this is archival replay. Validate source media fingerprint, each raw ASR observation SHA against manifest/exact source evidence, frozen audio crop SHA against actual original WAV if available, and model snapshot/revision against prior record. An input file's *declared hash* alone is not proof that the audio file bytes were checked; distinguish evidence surface explicitly.

Expected under exact previously uploaded FR inputs: 44 large-v3 lexical correspondence candidate blocks, 4 W03b displaced source sequences: c11 4w median CTC-minus-Whisper -5.925s, 5w +6.44s, 4w +11.43s, 13w +10.1s; c10 corroborating clean phrases near ~0.1-0.2s agreement. Repeated C2 does not mint a unique 4gram anchor; false subtitle-credit text from music/silence does not establish source coverage; W05a v3 omission must remain visible instead of fabricated. Any mismatch in exact inputs/metrics gets exact trace and explanation, not force-fit. No output candidate can set timing_accepted or transcription_accepted true. Preserve original source units, character-level Mandarin and all 41 untimed original ZH source utterances.

## 3. Strictly bounded direct-audio review / speaker-turn table

Reuse existing 18 archived clip records, host-local actual media and already available listening tools. Prioritize ~212-236s W02b child turn sequence (Tao/Zia), ~235-246s W02a displaced phrase, ~985-1014s W03b 4 multiword shifted matches, and W04a [~1200s,~1256s] split if a verified prior clip directly covers speech (otherwise extract one or two new bounded 10-30s clips). Do not widen into a full episode, new model download, forced-alignment checkpoint or generic diarization study. In particular, DO NOT rerun pyannote Community-1 on W02b merely to rediscover the previously proved collapsed single SPEAKER_00.

Independently audit original native audio. For every reviewed W02b acoustic turn, record actual event start/end, the bounded auditioned WAV crop SHA and separate review-record SHA, source French words independently heard, whether it is a distinct adjacent voice, and whether the identity of that voice is genuinely proven across turns. Whisper segment [220.22,221.66] contains both 'Toi ?' and 'Parfaitement !' while pyannote reports one speaker; DO NOT infer one voice from segment or pyannote. Where a distinct turn is heard but identity remains uncertain, mark boundary positive and identity unknown. Link exact source unit IDs where event ownership can be demonstrated, keep partial/displaced units explicit, never erase raw words or assign character names by text alone. For W02a/W03b confirm real speech interval with audio and distinguish true CTC displacement from lexical mismatch. Report W04a if positive evidence; unresolved cases remain unresolved.

Produce ONE private event review table (JSON plus short markdown) whose schema has: case, media SHA, crop SHA, audited start/end, transcription source and status, proposed source unit IDs/ordinals with original CTC and challenger time, audio_review_attested (boolean), speaker_change_attested (boolean), speaker_cross_turn_identity_status (proven|unresolved), and reviewer evidence hash. This is evidence for future Central-designed production integration, NOT a new accepted source v1 contract or automatic speaker override.

## 4. Exit and publication

No new full ASR/Whisper/Riva/Studio inference, no service restarts or GPU resource allocation. Reuse exact existing output for comparisons; run only bounded audio replay/inspection as needed. Publish ONE sanitized evidence-only commit to existing PR #15, with source identity, CPU/static result, 44-block match diagnostics with minimal bounded case samples (do not publish full copyrighted scripts/audio), W02b reviewed speaker-turn evidence summary, W02a/W03b reanchoring candidates, false-positive/missing and negative-control report, Mandarin control and model-adoption recommendation, runtime import proof, COMMAND_SHAPE_FRICTION, SHA256/bytes manifest. Keep complete raw challenger results and bounded audio review records in a self-contained private tar.gz with manifest/checksums, no huge media/model weights/tokens. Verify every public member SHA, byte count and final exact-head after push. No source/test changes, PR/issue/merge/deploy mutation beyond this one evidence commit.

Return:
 ISSUE14_INDEPENDENT_ANCHORS:
   RESULT=PASS|PARTIAL|BLOCKED
   ADMITTED_PR_HEAD=<fullSHA>
   VALIDATED_SOURCE_SHA=<fullSHA>
   VALIDATED_SOURCE_TREE=<fullSHA>
   CURRENT_PR_HEAD=<fullSHA>
   MAIN_SHA=<fullSHA>
   SOURCE_DIFF=<none|exact unapproved changes>
   CPU_FOCUSED_FULL_STUDIO=<counts/exits, source binding>
   ARCHIVE_PROVENANCE=<digest and 56/56 manifest matches>
   WHISPER_V3_ALIGNMENT_DIAGNOSTICS=<counts, c10/c11 W03b offsets, uncertain vs reliable, exact source/crop refs>
   TURBO_NEGATIVE_CONTROL=<false speech/repeats and omitted dialogue>
   W02B_AUDIO_REVIEWED_TURNS=<actual event timings, number distinct voices, identity confidence, per-turn unit links and SHA>
   W02A_W03B_W04A_REVIEW=<corroborated event/time vs model mismatch>
   FR_22_CASES=<changed/unchanged outcome references>
   ZH_EP166_CONTROL=<41 untimed/69 unmatched preserved>
   ASSUMPTIONS_AND_LIMITS=<WER/CER/DER UNVERIFIED, model agreement not truth>
   PUBLIC_EVIDENCE_ROOT=<exactSHA:path>
   MANIFEST_SHA256=<sha256>
   PUBLIC_MEMBERS_VERIFIED=<integer>
   PRIVATE_REVIEW_ARCHIVE=<path sha256 bytes>
   CENTRAL_NEXT_ACTION=<one concrete implementation decision>
   ISSUE14=OPEN PR15=DRAFT
 COMMAND_SHAPE_FRICTION:
   encountered=<yes|no>
   events=<[]|canonical events>
   omitted_sensitive_values=<yes|no>

If a genuine correctness failure occurs, preserve the reproduction and return PARTIAL rather than silently changing acceptance rules, creating another PR or delegating substantive architecture.