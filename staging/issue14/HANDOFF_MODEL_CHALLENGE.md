# Issue #14 / PR #15: one model challenge + source-fidelity architecture decision

LOCAL_AGENT_PRIMARY=build
REPO_ID=fvanevski/nemovox
ISSUE_NUMBER=14
PR_NUMBER=15
BRANCH=issue/14-longform-timing-boundaries
BASE_MAIN_SHA=8f6c03cbc70e84bf1970ba364207a02c9fe22b44
CENTRAL_SOURCE_SHA=753219a69dd9f05e861879e29e197898e3756b7c
CENTRAL_SOURCE_TREE=1c8c71eb828c57d27b6cf2a672b6b555609d4d97
DESIGN=staging/issue14/MODEL_CHALLENGE_DECISION.md
ARCHITECTURE=staging/issue14/ARCHITECTURE_REDIRECTION.md
EVALUATOR=evaluations/issue14/model_challenge/score.py
REFERENCE_PACKET=1bbcb3c8511e2bf2dfd5798d88d44789068549b5:staging/issue14/evidence/20261123T004035Z/
ACOUSTIC_PACKET=36fbf98021094b58efac5de76c7a3432f0a3a7f8:staging/issue14/evidence/20261123T000342Z/
MODEL_DOWNLOAD_ROOT=/workspace/models
RESULT=one evidence-backed model shortlist and next Central architectural implementation decision
NO_NEW_PR=yes; NO_MERGE=yes; NO_PRODUCTION_ACTIVATION=yes; NO_CODEX_REQUEST=yes

## Ownership and setup

You are the Build primary agent, running in a genuinely fresh task from the authenticated Nemovox checkout (normally /workspace/nvidia-transcribe). Agent-first: read AGENTS.md and design documents; safely fetch exact remote head, prove branch/PR/issue identities and ancestry to CENTRAL_SOURCE_SHA, create an isolated private worktree. Do not reset, clean, rebase, stash or overwrite the human's canonical checkout. If the branch has changed materially, STOP STALE and return full SHAs.

Central has authored the evaluator, fixed corpus/criteria and model-selection architecture. Your responsibility is host-bound runtime/model acquisition, test-only official SDK adaptation, measurement, and one complete evidence handback. You may fix deterministic test-only adapters and actual API/environment binding issues in a bounded private evaluation workspace without seeking another micro-handoff. Do not independently redesign production source reconciliation, ASR, global diarization, translation, ASS or GPU lifecycle. If the Central scorer fails because of a demonstrated bug, submit the smallest corrected source/test diff on this SAME PR with exact-head proof and rerun its tests; report the complete diff. An unsolved substantive algorithm/acceptance question returns to Central.

Current source/evaluation admission:
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m "not gpu" tests/test_issue14_model_challenge.py
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m "not gpu"
  git diff --check

Verify the actual Nemovox venv path, interpreter, source commit and source tree first. Reuse prior 308 source tests / frontend receipts if source/tests did not move; test newly staged evaluator and impacted tests at the current exact SHA. Do not claim PASS for new tests until they run.

## New user authority for testing models

The user authorizes downloading missing open models for testing, storing only under /workspace/models, and potential later incorporation after quality validation. This permission is NOT production adoption, broad package upgrades, or permission to evict unrelated GPU services.

Inventory and authenticate existing Nemotron 3.5 NVFP4, Qwen3-ASR-1.7B, Qwen3-ForcedAligner-0.6B, pyannote Community-1 and Silero VAD. Existing checkpoints should be reused.

Fetch the following absent challengers at their full pinned Hugging Face revision as needed, subject to actual disk quota and license access:
  FireRedTeam/FireRedVAD -> /workspace/models/FireRedVAD
  nvidia/Nemotron-3-Diarization -> /workspace/models/Nemotron-3-Diarization
  nvidia/parakeet-tdt-0.6b-v3 -> /workspace/models/parakeet-tdt-0.6b-v3
  FireRedTeam/FireRedASR2-AED -> /workspace/models/FireRedASR2-AED

Resolve Hub full revision first, then use the authenticated local Hugging Face downloader. The known shape is:
  hf download <repo-id> --revision <full-revision-sha> --local-dir /workspace/models/<stable-functional-name>
Verify hf download --help if the host CLI differs. Record exact repo revision, license, config, digest/size and acquisition status. Never overwrite an existing model in place, install weights in GitHub, fetch private datasets, or bypass a gated model's conditions. Keep tokens private. You may create a new pinned private evaluation-only venv outside /workspace/pyannote and the Nemovox production venv. Do not upgrade global CUDA, Python, system libraries, PyTorch, live NeMo or vLLM.

NVIDIA multitalker-parakeet-streaming-0.6b-v1 is NOT in this trial: its public card requires external speaker activity / one instance per speaker; its current French/Mandarin accuracy is not established, and its license is NVIDIA Open Model License, not CC-BY. Parakeet TDT v3 supports French but not Mandarin. Pyannote Community-1 is CC-BY-4.0 with HF access conditions, already cached locally. Standalone Nemotron-3-Diarization shares an encoder lineage with Nemotron 3.5, so its predictions are not independent human truth.

## Fixed test corpus and measurement ownership

Use the SAME independently host-auditioned original audio in both languages. Read the prior case/adjudication packets and authenticate every media fingerprint and crop offset before testing; prior model output, VAD, diar and approximate word counts cannot substitute for hearing the source. Corpus covers French H1 120–140, H2 360–380, H3 900–920, H4 920–940, c28 1099–1156, W04b/c 1230–1274, W05a/S05 1338–1379.901, W02b 209–232, W03a 985–1002, C1 686–700, C2 1022–1036, song 25–40, SFX 1200–1216, and remaining audited cases; Mandarin 96–114 and 518–536. Keep distinct actual speech, legitimate short repeats, overlap, songs, SFX and EOF speech. Use all applicable predeclared case windows, not cherry-picked successes. If annotations are insufficient, use the existing deterministic held-out selection and record provenance before inspecting new model output.

Private reference JSON format: schema nemovox-issue14-acoustic-reference/v1, media_fingerprint_sha256, regions:[{id,start_s,end_s,acoustic_truth:"independently_auditioned",events:[{start_s,end_s,kind}]}].
Private prediction JSON format: schema nemovox-issue14-model-predictions/v1, same media_fingerprint_sha256, model:{repo_id,revision,task} (task must be vad|diarization|asr|alignment), regions:[{id,status:"ok|not_run|blocked|failed|skipped_unsupported",observations:[{start_s,end_s,kind}], optional source_text}]. Every region must appear, even when blocked or unsupported. Times are ABSOLUTE MEDIA times, NOT excerpt offsets.
Kinds: speech, singing, lexical_shout, music, sfx, silence, nonlexical_vocalization, unknown. Preserve music/singing separate from dialogue. The Central scorer only calls something recognized if it temporally overlaps an independently heard event: it is NOT an audio-semantic correctness or DER oracle. Optional reference text requires source_text_verified=true, exact human-auditioned source_text, and language fr or zh, never text copied from the tested ASR. If exact reference text absent, WER/CER must remain UNVERIFIED. Do not publish copyrighted exact transcripts/audio.

Scorer command (workdir: authenticated isolated repo):
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python evaluations/issue14/model_challenge/score.py --reference <PRIVATE_REFERENCE_JSON> --prediction <PRIVATE_MODEL_JSON> --output <FRESH_PRIVATE_SCORE_JSON>
Never overwrite existing scored artifacts; use a new output path and exact model revision each time.

## Integrated measurements, sequential with owned resources

1. Activity/singing/music: compare Silero VAD v6 and pyannote Community-1 with FireRedVAD on the entire fixed corpus. Preserve FireRedVAD's multi-label categories. Key discriminators: H2 missed words, c28 shout/rescue behind SFX, W04c commands/narration, genuine theme vocals versus SFX false triggers, clean C1/C2. Report event-level hits, missed bands, speech-in-nonspeech overlaps and classifications. Never use model/VAD negative predictions to delete acoustically heard words.

2. ASR alternatives: compare archived Nemotron source observations with existing Qwen3-ASR-1.7B and downloaded Parakeet-TDT-v3 (French) or FireRedASR2-AED (Mandarin). Use genuine original clips, lexical omissions and hallucinations on quiet/SFX/music negatives. Run exact private WER/CER only with independently verified transcription, otherwise event and coverage comparisons. No 23-minute new ASR. Do not invent unsupported language scores. Do not use ASR-to-ASR agreement as truth.

3. Timing: use locally cached Qwen3-ForcedAligner-0.6B in one resident batched session against source-audited true French/Chinese short turns and an explicitly false W04c candidate/SFX transcription. Compare timing to independently labeled audible bands and record false text that still receives plausible alignment. Do not infer acoustic word existence from monotonic forced-alignment timestamps. Never launch per-character aligner subprocesses.

4. Diarization: compare pyannote Community-1, existing Nemotron/TitaNet and Nemotron-3-Diarization on W02b, S05, c28 and Mandarin 525s. Record actual separate voices, false merges/splits, overlap and chunk-to-chunk identity uncertainty; no assumed episode-wide speaker number. Preserve regular (overlap) and exclusive (display-oriented) pyannote outputs separately. Optional single whole-French-episode diarization comparison is allowed for Community-1 and standalone Nemotron-3 only when pilot results show relevance and host resource admission permits. No whole-episode ASR.

Each GPU task requires actual nvidia-smi headroom and the current Nemovox/host admission contract. If approved owned model service interfaces can safely admit it, use them. Otherwise use a verified isolated short offline model run with no live-service eviction. If neither route is safe, mark blocked for that model, not a reason to weaken ownership. Record sequential model load/release, peak allocated VRAM, wall-time, first-result latency and cleanup. Resource cost is part of model selection.

## Decision / stop criteria

Return a source-anchored judgment:
- Which detector contributes usable recall signals without becoming a hard audio-exclusion authority?
- Which French and Mandarin ASR challenger actually recovers previously omitted heard words without disproportionate hallucinations and latency?
- Whether forced alignment improves timing only when the underlying transcript is correct, and how false-text controls must be handled.
- Whether standalone diarization improves rapid alternations, genuine overlap, and cross-window identity.
- A single coherent next CENTRAL source implementation objective for AudioRegion + bounded source recovery + source-turn/word alignment/speaker evidence, including compatibility with v1 source and renderer lineage.

Do not turn this into a model shopping spree, generic ASR leaderboard, new test corpus, dependency modernization or one-heuristic micro-fix campaign. Finish once a complete, honest, comparable source-quality assessment supports or rejects each relevant challenger (with explicit unknowns where host capabilities prevented tests). This campaign is not Issue #14 completion.

## Publication: exact-source, complete public-safe packet

Publish ONE sanitized evidence-only commit on existing PR #15, after exact remote readback. Keep source/tests unchanged unless Central scorer's demonstrated bug needs a narrow corrected source commit, in which case fully requalify the new SHA before evidence publication. No force/rebase/merge/new PR.

Required under staging/issue14/evidence/<new-UTC-run-id>/:
  README.md
  source-identity.json
  validation/model-inventory.json
  validation/model-scorecard.json
  validation/source-coverage.md
  validation/timing-and-speakers.md
  validation/runtime-and-cost.json
  review/architecture-selection.md
  COMMAND_SHAPE_FRICTION.txt
  manifest.json

Manifest must include SHA256 and exact byte counts for all public members except itself. Check the exact Git staged paths and ignored files before commit: the previous campaign missed one *.jsonl due to .gitignore and needed force-add. Preempt that by using an explicit designated path manifest before the single publication commit, not by silently leaving a missing file. Publish the scorecard and bounded descriptors, not private WAV, full episode transcripts, raw model responses, weights, secrets or all logs. Keep private originals in fresh chmod-700 host scratch and return evidence-root digests.

Direct handback:
ISSUE14_MODEL_CHALLENGE:
  RESULT=PASS|PARTIAL|BLOCKED
  ADMITTED_PR_HEAD=<full SHA>
  SOURCE_SHA=<exact tested SHA and tree>
  CURRENT_PR_HEAD=<full SHA>
  MAIN_SHA=<full SHA>
  EVIDENCE_ROOT=<full SHA:path>
  EVIDENCE_MANIFEST_SHA256=<digest>
  SCORER_TESTS=<counts/exits>
  MODEL_DOWNLOADS=<pinned repo revisions, checksum and license; present/new/skipped>
  AUDIO_REFERENCE=<media hashes and independently heard labels provenance>
  FRENCH_ASR=<comparable WER if verified, omissions, hallucinations and unknowns>
  MANDARIN_ASR=<CER if verified, omissions, hallucinations and unknowns>
  ACTIVITY=<per-model missed speech, song/music/SFX false detections>
  DIARIZATION=<false split/merge/overlap and continuity findings>
  ALIGNMENT=<true and false text result/timing error bands>
  GPU_AND_HOST_STATE=<actual ownership, measured peak, nonmutation and cleanup>
  NEXT_CENTRAL_SOURCE_OBJECTIVE=<one cohesive implementation target>
  PUBLIC_MEMBER_INTEGRITY=<verified members/expected members>
  PRIVACY=<PASS|FAIL>
  SOURCE_DIFF=<none or exact full diff>
  BLOCKERS=<material only>
  PR15=DRAFT ISSUE14=OPEN

Canonical separate friction:
COMMAND_SHAPE_FRICTION:
  encountered=<yes|no>
  events:
    - scenario=<scenario>
      requested_shape=<sanitized requested invocation>
      effective_shape=<actual invocation/rejection>
      rewrite_provenance=<none|rtk_opencode_hook|shell_or_alias|other_harness_hook|unknown>
      event_kind=<pre_execution_rejection|execution_failure|successful_recovery|guard_annotation|harness_rewrite|runtime_misbinding|interface_discovery|other>
      execution_effect=<not_executed|executed_failed|executed_succeeded|recovered_after_retry|unknown>
      correction_or_annotation=<bounded fact>
      replacement_shape=<corrected shape|none>
      recipe_reference=<GHDEV-10 key|none>
      resource_reference=<source|none>
      contract_conflict=<yes|no|uncertain>
      durable_gap_candidate=<yes|no|uncertain>
  omitted_sensitive_values=<yes|no>
If none, encountered=no, events=[], omitted_sensitive_values=no. Record friction separately from substantive model quality.
