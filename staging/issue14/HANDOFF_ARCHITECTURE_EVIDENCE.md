# Issue #14 / PR #15: ONE decision-focused acoustic-truth and reconciliation evidence campaign

## Goal and boundaries

Deliver the **minimum new, independent evidence** required for Central to replace the increasingly overlapping boundary/CTC/speaker heuristics with a simpler audio-region/observation-centered approach. This is NOT another c28/W04b/W05a ASR-window search and is NOT authorization to design or implement the new architecture locally.

REPO_ID=fvanevski/nemovox
BRANCH=issue/14-longform-timing-boundaries
ISSUE_NUMBER=14
PR_NUMBER=15 (OPEN/DRAFT)
SOURCE_SHA=4c32acf659891fe999d2ae6e15f59ccb5750d92f
SOURCE_TREE=a4e866f53899fe159c99793ff8a78326a10b26f5
MAIN_BASE_SHA=8f6c03cbc70e84bf1970ba364207a02c9fe22b44
PREVIOUS_PROBE_EVIDENCE=11a0ab8bfa0a9be4c67fcc2b822cd529abce5461:staging/issue14/evidence/20261122T231624Z/
BASELINE_EVIDENCE=5bf6f37d173cfcc160fdf03b60b051cd0f5e7dea:staging/quality-audit/mcog-s04e01/evidence/20261122T205259Z/
FULL_RUN_EVIDENCE=a1812212f6e5c7a82bcd0a27769130fda420a3d3:staging/issue14/evidence/20261122T225739Z/
CONTROLLING_DESIGN=staging/issue14/ARCHITECTURE_REDIRECTION.md
EVIDENCE_ROOT=staging/issue14/evidence/<UTC-run-id>/
EXECUTION=OpenCode build, with actual multimodal/acoustic-listening capability only if verified
GPU_INFERENCE=forbidden (reuse existing 16-chunk, 35-chunk, 9-probe evidence)
EVIDENCE_REMOTE_UPDATE=allowed (one non-force sanitized evidence-only commit, same branch)
SOURCE_REMOTE_UPDATE=forbidden
EVIDENCE_PACKET_REQUIRED=yes
EVIDENCE_ARCHIVE_REQUIRED=no unless indispensable private acoustic reference cannot be conveyed via safe summary
PRIVATE_ARCHIVE_UPLOAD_REQUIRED=no by default
PRODUCTION_MUTATIONS=forbidden
NEW_ISSUE_OR_PR=forbidden
MERGE_REVIEW_REQUEST_DEPLOYMENT=forbidden

Use a genuinely fresh, isolated worktree/scratch root. Read current AGENTS.md, the current design memo and the prior evidence packets, then authenticate remote branch, PR #15 draft, main base, exact SOURCE_SHA and source tree. Previous evidence carrier and design docs may be later commits with identical \`src/tests/web\`. Do not alter /workspace/nvidia-transcribe original production checkout, the original Studio job, media, model services, source VENV, or previous private data. Do not load models, change prompts, start Riva/Qwen/ASR, perform another 23-minute run, generate subtitles, or install packages. Existing ffmpeg/ffprobe and offline Python scripts for extraction/analysis are allowed.

Host source candidates: original media
  /opt/mcog/Les mystérieuses cités d'or Saison 4 Episode 1 [YlVGUDNbyXk].webm
original job
  /workspace/nvidia-transcribe/outputs/studio/jobs/980af31f014245c4b200e0af8b33ebca/
prior pilot/full runs and nine-probe private scratch must be discovered and authenticated from their own manifests, not guessed; absent files are explicit gaps, not permission to re-infer.

## Phase A: authenticate artifacts and normalize prior observations (CPU only)

Verify media/recording hashes from prior source-identity.json, current/production separation, source commit/tree, and all original 16 raw responses, 35 pause-aware raw responses, 9 probe responses if still present. Read-only. Bind source offsets to original media timeline. Produce an immutable index keyed by [media hash, excerpt start/end, model/request hash, raw-response SHA], not \`chunk_014\`, \`utt-0138\`, \`spk:0\` or current speaker IDs alone.

Construct for the priority and control regions an observation-comparison grid: old fixed window, pause-aware run and existing nine alternate windows; record presence of speech hypothesis, lexical count, CTC plausible chronology, raw diar interval, attributed local speaker tag, finish state, overlap coverage and any direct contradiction. Do NOT treat cross-model self-agreement as independent ground truth; model diarization and CTC are correlated Nemotron outputs.

## Phase B: independent acoustic labels (the critical missing evidence)

Read-only extract bounded 16-kHz WAV excerpts into a fresh chmod-700 local root. Do not publish WAVs, full transcriptions or copyrighted raw text. Required regions with distinct purposes:

- **A / c28 1099–1124s and 1123–1156s:** MOST IMPORTANT. Independently listen for actual dialogue versus song/music/SFX/silence across the previously unverified 1100–1123s interval; record speech events, source start/end bands, number of turns, distinct voice evidence, repeats, and approximate recognized-word coverage per subregion (1099–1110, 1110–1123, 1123–1144, 1144–1156). If short clips are needed for perception, split without cutting in-progress words. If exact wording is important to adjudicate a mismatch, keep it in a private labeled reference with SHA and publish only coverage/counts/limited safe descriptors.
- **B / 1230–1274s:** Independently distinguish W04a, W04b and W04c event times, separate real turns, and whether apparent moved CTC words have acoustic counterparts; note music/SFX areas. Do NOT assert the 1242–1274s ASR candidate is correct solely because it is internally monotonic.
- **C / 1338–1379.901s:** Independently label W05a child utterance, narrator transitions and S05 Pichu voice, including genuine overlapping/repeated words and end-of-media tail; compare 5-word vs 6-word ASR alternatives to audio.
- **D / 209–232s and 985–1002s:** Real alternating/overlapping voices and already known displaced duplicate across two chunks. Verify whether a proposed event-level correspondence can preserve distinct simultaneous speech.
- **Controls / 686–700s C1, 1022–1036s C2**, plus 25–40s theme song and 1200–1216s largely SFX/music: characterize how acoustic detectors behave on clean speech, real repeats, singing, music and non-speech noise.

Include **four held-out ~15–25s source regions** selected deterministically before checking their corresponding ASR outputs, excluding the named priority/control bands. Derive a reproducible SHA-256 seed from the known video digest + literal "issue14-heldout-audio-v1" and select four distinct non-overlapping candidate intervals in the 23-minute episode. Record selection code, eligible interval grid, exclusions and chosen times. Aim for a mix of naturally occurring activity but DO NOT change the seed/selection to produce favorable examples. If authenticated private Mandarin EP166 audio and response evidence already exists, add at most TWO distinct Mandarin windows (one overlap, one ordinary) as a cross-language *check*, not a second investigation. Absence is recorded, not grounds to download media.

**HARD EPISTEMIC RULE:** A transcript or diarization returned by Nemotron is NOT audio listening. If your harness cannot truly perceive the short WAV, do NOT claim HEARD/NOT_HEARD. Use an actually available multimodal-audio inspection tool if it supports real listening; otherwise mark ACOUSTIC_PERCEPTION=UNAVAILABLE, generate a bounded local reference index and clear private operator hearing task, and return PARTIAL. Do not invent phonetic/semantic labels from CTC/ASR. Independent basic energy/silence is activity only, never proof of spoken words.

## Phase C: offline detector and decision calibration, no new ASR

Across **all 35 archived chunks**, derive candidate regions from model diar spans with no lexical CTC support, non-monotonic word chronology and chunk-overlap disagreements. Count distributions (including how many >=1.0s unpopulated spans), their positions and whether they coincide with original core or only inference context. Compare candidate triggers to the independently heard priority and four held-out windows. Include false-positive music/SFX and false-negative speech omissions. If a source label is unavailable, leave undecided; NEVER report whole-episode precision/recall based on chosen defect-enriched regions.

Do an offline threshold-free **counterexample matrix**: (i) complete-looking \`stop\` yet poor coverage; (ii) real multiple voices vs same local speaker tag; (iii) same lexical phrase in two genuine source events; (iv) same source event in two distinct ASR observations shifted in CTC; (v) recorded singing/SFX with diar detections; (vi) source-true end-of-file speech. Distinguish proof of *one acoustic event* from agreement of ASR hypotheses.

Compare two conceptual ownership policies with source-event labels, **without implementing either production algorithm**: (1) existing core-ownership+heuristics and (2) an event-centered correspondence that groups all source-overlapping candidates and defers selection until evidence supports one candidate. For each hand-annotated region, give counts of preserved, omitted, duplicated, unresolved events and intervention/retry candidates, identifying minimal positive evidence needed to automatically select. Track whether the event-centered alternative could operate with only existing audio activity + raw Nemotron responses or would actually need an independent VAD/aligner. A source-record structure or spreadsheet calculation is okay; do NOT code a local candidate reconciler, retune thresholds, or assert that the new design passed production.

## Phase D: pause planning and speaker evidence useful to a compact replacement

Extract the **existing** 87 ffmpeg silence intervals and 35 chosen boundaries (if authenticated private plan exists), mark each chosen pause/forced cut's position relative to independently labeled speech events and short clips with bad initial/end context. Produce concise detector statistics and numeric inputs for a future Central-authored global cut optimizer (duration targets, silence duration, cut proximity, inferred covered activity). Do not implement a new planner or invent an \`>=8s\` context rule.

For W02b/S05 and held-out genuine overlap, record separate raw diar channels vs LLM speaker-tag maps and independently heard turn count. Tag local IDs explicitly as request-local. Do not infer an episode-wide true speaker count or tune TitaNet cosine threshold based solely on known 0.65/0.70 examples.

## Publication and acceptance evidence matrix

| Criterion | Required artifact | Observation | Privacy |
|---|---|---|---|
| Exact-source input identity | source-identity.json | full Git refs/tree, media+job+private raw digest readbacks, host services unchanged | public sanitized |
| Distinct independent truth labels or explicit lack thereof | validation/acoustic-event-ledger.jsonl, validation/perception-method.md | per-region source events, acoustic status, timestamp error bands, voice separation & direct hearing method | public minimal; raw labels/audio private |
| Generalized coverage trigger viability | validation/detector-contingency.json, validation/candidate-observations.jsonl | all 35 raw chunks, unpopulated diar/CTC issue distribution, sampled true/false positives and unknowns | public counts/hashes only |
| Overlap/source-event selection design | validation/event-correspondence-matrix.md | current vs alternative conceptual handling of duplicate/omitted/real repeated events and uncertainty | public |
| Planner boundary evidence | validation/pause-boundary-inputs.json | 87 pause intervals + 35 cut times when available; 64.241 example contextualized | public |
| Fixture authority mapping | review/fixture-adjudication.md | keep/rewrite/retire the observable requirements; do not promote internal heuristic values to truth | public |
| Reproducible control and privacy | README.md, COMMAND_SHAPE_FRICTION.txt, manifest.json | sampling seed, exact method, member hashes/sizes, no production/source changes | public |

Packet: \`staging/issue14/evidence/<UTC-run-id>/\`, at most one public-safe evidence-only commit, all 7 required named artifacts above plus README/friction/manifest, exact SHA-256 and bytes. No full copyright transcript, WAV, original raw ASR responses, sensitive hosts/paths, model secrets, or large logs. Store private raw support under chmod-700 collision-safe scratch, optionally provide a separate securely attachable bounded archive ONLY IF exact source-words or audition is indispensable for Central review and cannot be communicated safely. If audio perception is unavailable, the public evidence packet must expose that gap and link a local PRIVATE cue index; it must not claim acoustic verification.

Read latest ref before publication; prove only designated packet paths changed; one non-force CAS/normal push, verify remote commit exact parent/head/tree and all member SHA+bytes. If concurrent drift, fail closed / return local packet; no rebase/reset/force. Source/tests/Studio/inputs/workers/model units and user worktrees remain unchanged. Do not run full CPU test suite because there is **no source modification**; use Python syntax and deterministic checks for only private analysis scripts, if any.

LOCAL_AGENT_HANDBACK:
  RESULT=PASS|PARTIAL|BLOCKED
  SOURCE_SHA=4c32acf659891fe999d2ae6e15f59ccb5750d92f
  SOURCE_TREE=a4e866f53899fe159c99793ff8a78326a10b26f5
  MAIN_SHA=<fresh full sha>
  EVIDENCE_CARRIER_SHA=<fresh full sha>
  SOURCE_AND_PRODUCTION_INVARIANCE=<pass/fail>
  ACOUSTIC_PERCEPTION=<method+actual independent listening|UNAVAILABLE>
  C28_1100_1123_SPEECH_TRUTH=<source event counts/presence/uncertainties>
  C28_1123_1156_TRUTH_AND_C29_OVERLAP=<independent vs model agreement>
  W04B_W04C_TIMELINE=<independent acoustic truth>
  W05A_S05_SPEAKER_AND_TAIL=<independent acoustic truth>
  HELDOUT_SELECTION=<seed derivation+chosen intervals+no post hoc selection>
  COVERAGE_TRIGGER_CONTINGENCY=<labeled positives/negatives/unknown; avoid episode-wide rate>
  EVENT_CORRESPONDENCE_DECISION=<which alternative is justified, what remains open>
  PAUSE_PLAN_DATA=<interval count, cut count, identity and evidence>
  MANDARIN_CROSS_LANGUAGE=<available|unavailable; limited evidence>
  EVIDENCE_PACKET_REQUIRED=yes
  EVIDENCE_REQUIRED_ARTIFACTS=<manifest paths>
  EVIDENCE_INCLUDED_ARTIFACTS=<verified paths>
  EVIDENCE_MISSING_ARTIFACTS=<none|names and reasons>
  EVIDENCE_PRIVACY_CHECK=PASS|FAIL
  EVIDENCE_INTEGRITY_CHECK=PASS|FAIL
  EVIDENCE_GITHUB_ROOT=<full commit:path or BLOCKED>
  EVIDENCE_ARCHIVE_REQUIRED=<yes|no with cause>
  PRIVATE_ARCHIVE_ADDS_REQUIRED_EVIDENCE=<yes|no>
  PRIVATE_ARCHIVE_UPLOAD_REQUIRED=<yes|no>
  CENTRAL_REVIEW_ATTACHMENT_REQUIRED=<yes|no>
  COMMAND_SHAPE_FRICTION=<encountered=no/events=[] or exact events>
  BLOCKERS_AND_UNCERTAINTIES=<real gaps only>
  ISSUE14_PR15=OPEN/DRAFT
  REQUIRED_OPERATOR_ACTION=<NONE or exact needed action>

PASS means *the evidence-gathering procedure and data classification are complete*, NOT that a new reconciliation implementation was tested or that the prior output is source-correct.
