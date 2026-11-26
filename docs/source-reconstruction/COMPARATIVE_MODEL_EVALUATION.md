# Bounded comparative source-model evaluation (#24)

## Authority and outcome

Operator approved this evaluation before authoring another production recovery
mechanism. This is host-bound comparative research, not implementation acceptance
or a model-replacement decision. Central owns selection, architecture, production
code, tests and promotion. The qualified production-code baseline is
4e813da48b2193d13190e0d200f75dbd1b91b35d (tree
4d05dee80405a967dc1db012925de49bfd864ab7), main
92d9f1bde9eae00e467bfd0d8e804599c60ad87b. The subsequent protocol-only commit
is the execution instruction identity; prove its delta contains only this file.
This protocol supersedes further live recovery qualification instructions for
this evaluation, not the issue acceptance criteria.

AGENTS.md and docs/model-resources/LIFECYCLE.md govern. Build/host experiment
role: prepare detached checkout, disposable evaluation environments, thin
upstream API invocation/export scripts and host evidence. No production edits,
new adapters, canonical selection algorithms, translation policies, services,
registry changes, PR, merge, deployment or issue closure. Local scripts may
decode, invoke, serialize, normalize and calculate specified metrics; do not
assign unobserved times or implement candidate recovery policy. Return actual
adverse results. Central will choose what to implement.

## 1. Fixed evidence and inputs

Owner /workspace/nvidia-transcribe. Private root beneath its outputs/handbacks:
use umask 077 and mktemp -d with prefix model-comparison-<head8>.XXXXXXXX;
no pre-clean, reset, destructive cleanup or owner checkout switch.

Authenticate these existing archives (full digest, safe member paths and
internal checksum inventory before use):
- outputs/handbacks/source-robustness-failed-ctc-4e813da.Aro8jC4G/nemovox-issue24-failed-ctc-4e813da-review.tar.gz
  ed2302e69044bfb5425de511ffcfcf3e17ec601488a63300c9cb6ef8f2dbea77
- outputs/handbacks/source-robustness-resume-d0d4af0d.PrZ3WGAh/nemovox-issue24-successor-d0d4af0d-review.tar.gz
  6a729732cbe0840030ae303a528ef5f27216a974fa676be3e3e15aaeeea26661
- outputs/handbacks/source-robustness-intake-d3124a5f.CpjjR3LR/source-robustness-intake-d3124a5f.tar.gz
  86c63cceb96629a725126926ea1c20b92ea8a7f3e803e30414d960946a2a6c44
- outputs/handbacks/source-robustness-ordered-56cd120.frKi6CZn/nemovox-issue24-ordered-56cd120-review.tar.gz
  89836fc2b78d9cd4972f52aa387f2370bb475102a1d491e9ac7f90f4bdea3a5a

All paths relative to owner. Read latest archive admission.json,
mandarin-arms-analysis.json, arm A/B canonical-source.contract.json,
source-quality.json, translated.contract.json, arm B asr raw chunks/ledger,
and french-equivalence.json.
Use existing reports at original execution identities, not as fresh outputs.
The old analysis summary contains an inconsistent Arm B denominator; independently
count distinct selected IDs from the actual recording/ledger (378 selected,
318 localized, 303 failed-core units, 243 recovered there). Arm A 248/402;
French at 56cd120 qualified with scoped equivalence. No full-suite repetition
for a documentation-only commit; verify production tree equivalence.

Freeze corpus.json before exposing new recognition outputs:
D1: exact Mandarin 119s discovery clip
outputs/handbacks/source-robustness-resume-d0d4af0d.PrZ3WGAh/mandarin-76-195.mkv
SHA256 94b7d739d07c4a72557577688b59e7e5df6a36aa214e887a5bf7c64ccf8d5743,
source stream audio ordinal 0, parent interval 76–195s. Keep clip clock 0–119.
F1: original French S04E01 media identified by intake, full media SHA256
0e77ce3c809e498b97ffe48b53575c29e10d8b08a039b78e2150cd0c845c81b0,
audio ordinal 0 (actual French regardless of metadata), intervals [165,265]
and [975,1005] seconds. Includes known echo and recovered exposition plus
neighbor dialogue. Treat separately, not concatenated for inference.
H1: one independent 90–120s Mandarin excerpt, different episode/recording
than D1, selected from accessible local personal media before any new inference,
prefer multiple speakers, ordinary dialogue, some music/silence; avoid selecting
by candidate outputs. Record objective selection rule, parent fingerprint,
stream and exact interval, why independent, reference availability.
If none accessible, record blocked H1; do not silently substitute D1.
Total unique audio <=369s. No full episode, other languages or broad corpus.
Existing completed same-input observations are comparisons with their original
geometry; never claim a matched comparison where geometry differs.

Decode a single canonical 16kHz mono signed 16-bit PCM WAV for each clip from
the chosen stream with exact decoded-sample offsets, not keyframe-copy seek.
For example for bounded interval from a parent, use ffmpeg -v error -nostdin
-i "$MEDIA" -map 0:a:0 -ss "$START" -t "$DURATION" -ac 1 -ar 16000
-c:a pcm_s16le "$WAV" (adjust actual audio ordinal only if proven).
Hash complete WAV and decoded PCM; probe sample rate/count/duration; record
clip-relative clock and absolute parent offset. Do not include WAV/media in
handback. Supply each comparable model identical WAV bytes; where model
requires chunking, use fixed nonoverlapping cores <=30s for non-fused models
requiring short windows, optionally <=5s context on each side within actual
input, document offsets and retain all raw crop outputs. No inferred boundary
deduplication or retiming. Fused models may take whole clip; label geometry
as architectural variable, not a controlled same-crop comparison. Preserve
cross-boundary observations, not hidden output concatenation. For forced
alignment use exact original native inference WAVs/geometry, not clipped
canonical fragments with mismatched text; authenticate original crops using
manifest/ledger or reproduce exact PCM from parent.

## 2. Model admission and isolation (CPU/setup first)

Allowed model set, in priority order:
- Existing Nemotron NVFP4 (no download or modification).
- Qwen/Qwen3-ASR-1.7B and Qwen/Qwen3-ForcedAligner-0.6B.
- OpenMOSS-Team/MOSS-Transcribe-Diarize.
- FireRedTeam/FireRedASR2-AED (Mandarin recognition only).
- Existing Faster-Whisper baseline and one existing, locally available dedicated
  diarizer (prefer already qualified pyannote Community-1 or existing pyannote
  checkpoint; alternatively existing Nemotron-3-Diarization). No new dedicated
  diarizer acquisition is needed in this round.

First locate existing caches/checkpoints, verify all required assets and actual
revision/runtime. New downloads, if needed for named Qwen/MOSS/FireRed models,
are explicitly scoped evaluation-only acquisition under operator-approved
comparative evaluation; no automatic production-loader download or AGENTS.md
lifecycle exception. If local policy requires separate explicit acquisition
approval, report missing repo/revision/bytes and request that approval, continue
cached arms. Resolve immutable HF commits and upstream code revisions first,
inspect licenses, model config, required files, dependencies and custom code.
Use pinned official snapshots in a separate evaluation cache, e.g.
hf download "$MODEL_ID" --revision "$MODEL_SHA" --local-dir "$SNAPSHOT";
verify total size and free disk before acquisition, reuse verified existing
weights, no redownload of Nemotron. Do not download other variants or the
entire FireRed system if the AED-only interface suffices. No arbitrary
unreviewed remote code; inspect pinned custom model/processor code prior to
execution, use no secrets or privileged execution. No external audio upload.
During inference use local absolute model paths, offline flags and check
non-HF download routes; no implicit fallback. Official dependency installation
in evaluation-only venvs allowed, not system/owner/pyannote venv modification.
Use uv venv --python 3.12 "$PRIVATE_ROOT/envs/<model>" and uv pip install
--python "$PRIVATE_ROOT/envs/<model>/bin/python" <pinned compatible packages>
as applicable (reuse a compatible isolated existing environment only with
identity proof and no mutation). Verify CUDA sm_120 and actual device/dtype,
not just import. Prefer straightforward eager/SDPA Transformers local inference
over new inference servers, flash-attn builds, quantization or serving campaigns.
Allow one bounded dependency or batch-size correction per failed arm; ordinary
diagnosis and upstream-example adjustment allowed. Record exact lock/freeze,
commands, runtime and errors. No silent CPU fallback.

Use live nvidia-smi, process and lifecycle status, verify cooperative lock and
idle owner job before GPU work. Verify the actual cooperative lease contract;
coordinate temporary GPU processes under that same ownership/exclusion
boundary, without recursively acquiring a lock already held by a managed
pipeline. Use existing owned lease/lifecycle only;
do not stop or restart unrelated workloads or Studio. Run model groups
sequentially. Unload/exit one before loading next, except jointly admitted
Qwen ASR+aligner in a single process if resources fit. Reuse one loaded model
across all admitted clips and batches. If memory inadequate, batch down or
unload ASR before aligner, not evict others. Use actual measured available
VRAM, not parameter-count estimates. No forced device change. Monitor total
and per-process GPU usage, record baseline/peak/after, wall and model load
separately. Stop on unsafe state, ownership loss, unexpected acquisition or
unrecoverable OOM; isolated arm errors do not block others. Restore only
campaign-owned resources, release lock, prove final readback.

## 3. Execute fixed comparison matrix

N = current Nemotron reference (reuse D1 A/B and F1 retained receipts).
Q-align = frozen native transcript + independent Qwen forced alignment.
Q-ASR = Qwen3-ASR-1.7B source recognition + same Qwen aligner.
MOSS = native source transcript + segment times + anonymous speakers.
FireRed = Mandarin-specialized recognition, timestamps if actually returned,
otherwise Qwen aligner for optional timing comparison.
Whisper = existing independent observations, not a new large crop sweep.
Diarizer = independent speaker evidence, one existing checkpoint only.

D1: N, Q-align, Q-ASR, MOSS, FireRed, retained Whisper and available diarizer.
F1: N, Q-ASR, MOSS, available diarizer; Q-align only on the failed/quarantined
exposition evidence if a coherent exact audio/text pair can be authenticated.
H1: Q-ASR, MOSS, FireRed, available diarizer and one ordinary Nemotron primary
recognition run using existing CLI/ASR path; no downstream recovery/MT campaign.
If new N H1 unavailable due resource admission, report missing comparison.
Use the inspected public ASR-only command from the detached candidate root,
with the owner interpreter and actual existing resource configuration:

    PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m nvidia_transcribe.cli process-media "$H1_WAV" --output-dir "$PRIVATE_ROOT/nemotron-h1" --source-language zh --mode speaker --chunk-policy fixed --chunk-seconds 90 --context-seconds 10 --audio-stream 0 --speaker-embeddings none --alignment-fallback off --model-resources "$RESOURCE_CONFIG"

Resolve RESOURCE_CONFIG from the existing effective NEMOVOX_MODEL_RESOURCES or
host configs/model-resources.local.json, without exposing credentials. Compare
actual mode and chunk geometry to retained primary manifest. Do not hold an
external lease across process-media's own managed ASR acquisition; release the
experiment lease after proving its processes exited, and let the CLI acquire
its normal lease. No production patch. For N use actual ModelOpt mixed,
V1 runner, CTC middleware/trust-request-mm-kwargs and explicit CTC capture per
AGENTS.md. Retain raw CTC error, diarization and text.

Order: Q-align on retained failed primary text (load aligner once, batch lists
of exact native crop audio/text pairs, one pair per actual native request;
use parser's raw turn texts in original order, omit speaker markup but retain
turn/lexical mapping; no correction, no fabricated local timing, no
forced alignment of each residual row). Compare Arm A and B failed primary
crops separately; retain successful B chunk as control. If >5 min split only
at actually known coherent boundaries, otherwise record unsupported.
Q-ASR on D1/F1/H1, use recognition independent of native text, align own output.
MOSS on whole admitted clips. FireRed D1/H1. Diarizer once per clip if existing
resource admitted. All models source-language only, no prompt/hotwords derived
from target transcript, no reference-conditioned recognition. Use default
documented source-language transcription/diarization instruction, record
language hint and decoding parameters. Prefer deterministic decode where
supported. One ordinary pass per arm/clip, one additional shifted/smaller
bounded comparison on discovery if necessary to resolve material ambiguity or
truncation, total <=2x unique input audio per new recognizer excluding one tiny
smoke and forced-alignment-specific pairs. No repeated tuning on H1. No broad
hyperparameter sweeps. Do not initiate all-model end-to-end English review.

Verified upstream entrypoints (recheck pinned version and imports before use):
- Qwen: https://github.com/QwenLM/Qwen3-ASR#python-package-usage
  qwen-asr; Qwen3ForcedAligner.from_pretrained(local_aligner,
  dtype=torch.bfloat16, device_map="cuda:0"); align(audio=list_of_paths,
  text=list_of_texts, language=list_of_language_names). Record each raw item
  text/start_time/end_time. Qwen3ASRModel.from_pretrained(local_asr,
  dtype=torch.bfloat16, device_map="cuda:0", max_inference_batch_size=4,
  max_new_tokens=2048, forced_aligner=local_aligner,
  forced_aligner_kwargs=dict(dtype=torch.bfloat16,device_map="cuda:0"));
  transcribe(audio=paths,language=names,return_time_stamps=True).
  Use Chinese/French explicit language, record actual detected language where
  returned. If joint fit fails, transcribe without aligner then batch align
  the exact generated texts. Report recognition and alignment time separately
  where measurable; no guessed split for a joint API. Verify token limits
  and truncation; not an arbitrary 2048-token completeness guarantee.
- MOSS: https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize
  https://github.com/OpenMOSS/MOSS-Transcribe-Diarize
  Pinned upstream Python package/inference_utils:
  build_transcription_messages(local_wav), generate_transcription(model,
  processor,messages,max_new_tokens=4096,do_sample=False,device=device,
  dtype=torch.bfloat16), parse_transcript(result["text"]).
  Load pinned local snapshot via AutoModelForCausalLM/AutoProcessor,
  trust_remote_code only after code inspection, CUDA explicit, SDPA.
  Retain raw generation including invalid segments and stop/token status.
  Native intervals are segment times, not word times. Preserve anonymous
  speaker labels within each inference; no implied global mapping between clips.
- FireRed: https://huggingface.co/FireRedTeam/FireRedASR2-AED
  https://github.com/FireRedTeam/FireRedASR2S
  Use pinned upstream AED-only inference entrypoint with explicit local
  checkpoint and GPU, no assumption that system default config is AED or
  that all four system modules are required. Capture exact public invocation,
  native output and timing capability; no invented timestamps for text-only
  results. If no usable AED-only path without expanding acquisition, mark
  arm blocked and finish others.
These are interface projections, not a claim that host packages are installed
or these examples have executed on the host. Thin wrapper implementation
and version-specific mechanical corrections belong to local execution.

## 4. Common output and assessment contract

For each arm/case retain original UTF-8 output and a lossless normalized JSON
view: case_id, arm_id, actual input/crop hash, offset, requested/effective
language, response/row IDs, raw source text, source start/end (null when absent),
timing_kind (native_word/native_segment/forced_alignment/none),
speaker_label (null when absent), label_scope (request/crop/clip/unknown),
word/character observations with native granularity, invalid/deferred reasons.
Record both crop-relative and clip-relative values; derive parent times from
offset. Preserve non-speech annotations separately. No fabricated grouping,
unit timing, translation or cross-crop speaker assignment. Word-only output
may be displayed as observed word intervals; a diagnostic source SRT only
when native segment bounds or exact contiguous observed word envelope
support it. Any deterministic diagnostic grouping must retain constituent
indices and not invent times or imply a sentence boundary as a speaker turn.
No production canonical/translated contract generated by the wrapper.

Evaluate:
- D1 material clauses and actual localities: opening punch/exposition; fight
  clause around 45–50s; missing 59.92–72.32s; future ceremony 72–78s
  (source duration meaning); announcement 78–81s; repeated longevity
  84–92s; missing 103.88–108.98s; tail. These are inspection targets, not
  hardcoded desired wording or accepted acoustic truth. Compare all actual
  text and intervals, not just successes; investigate watermark-like text
  from prior Whisper 53.5–74s crop.
- F1 echo phrase at ~170–179s versus ~217–258s, Ophir exposition around
  990–996s, clean neighbors and speaker changes. Real repeats must survive.
- H1 assess full bounded excerpt without tuning against its output.
- Frozen pre-output reference: if existing independently authored matching
  source subtitles/transcript are available, verify actual audio locality,
  provenance, coverage and translation status; never use English subtitle
  backtranslation as exact source reference. Use bounded audio review for
  consequential cases if genuine human qualified listener/reference is
  available; otherwise explicitly mark reference unavailable. Do not ask
  operator to transcribe entire clips or claim to have listened if the agent
  cannot. Model agreement, native diarization union, generated captions and
  aligner confidence are proxies, not ground truth. Uncertain cases remain
  uncertain. Freeze reference identity and normalization before scoring.
- CER/WER only for authenticated reference spans, report numerator,
  denominator, substitutions/deletions/insertions, normalization (Mandarin
  simplified/traditional mapping if used; French accents/apostrophes;
  punctuation removal) and exact scoring spans. Otherwise report proxy
  text agreement/disagreement with no accuracy claims.
- Timing: measured duration and union of supported event intervals,
  invalid/out-of-crop/nonmonotonic/zero-length/missing bounds, gap lengths,
  overlaps (not automatically errors), duplicate occurrences at competing
  positions, actual silence/music behavior, boundary drift on matched
  observed events. Coverage denominator is clip or observed activity union,
  not known speech recall unless independently annotated. Forced alignment
  of wrong text does not prove occurrence. No credit for coerced times.
- Speaker: local turn separation, observed overlap, fragmentation/label
  reuse and continuity across natural speaker returns. Compare independent
  diarization evidence only as a proxy unless reference annotations exist.
  DER/JER or speaker-attributed WER/CER only with suitable reference,
  documented collar/overlap policy and permutation handling. Do not force
  cast size or invent speaker labels for Qwen/FireRed; independent diarizer
  timings are separate evidence, not an implemented fusion algorithm.
- Efficiency: model cold load, warm inference, alignment, total actual
  audio processed incl context, request count, wall, peak device/total VRAM,
  failures/retries, actual output completeness; normalize by unique input
  duration, not by processed repeated seconds. Compare ASR-only with
  ASR-only, not with baseline 338/466s including translation/review.
  Distinguish estimated full-episode scaling from measured short-clip cost.

Return per-case clause/interval evidence table with original alternatives,
status (supported/variant/omitted/disputed/unverified), evidence IDs and
reference authority. Include observed negative controls, not manufactured
examples. Select no winner by weighted arbitrary score or download count.
Central decision criteria: useful dialogue coverage and fidelity first,
valid timing/no material invented speech, meaningful speaker separation,
reasonable resource cost and no regressions in French/independent control.
Report if evidence supports timing-only repair, alternative recognition,
fused event reconstruction, or insufficient evidence. Do not promote model.

## 5. Handback (one private archive)

Required evidence matrix (all private, host-attested unless Central verifies):
- admission.json: exact staged SHA/tree, baseline delta, owner/remote identities,
  source imports if used, input archive checksums, host/lease state.
- corpus.json: frozen selection, media/PCM hashes, clocks, stream, reference
  provenance and selection before candidate exposure.
- models.json: resolved snapshots/code revisions, asset inventory/hashes,
  license, source URLs, package/runtime versions, device/dtype, actual
  acquisition, blocked arms; no tokens or weight bytes.
- runs/<arm>/<case>/: actual command/request settings, sanitized raw structured
  recognition/alignment/diarization outputs, normalized view, raw output
  digest and omission reasons, exits, error excerpts and timing/VRAM receipts.
  Include exact runnable thin evaluation scripts and environment lock/freeze,
  with secrets/absolute private names sanitized where nonmaterial.
- comparisons.json + HANDBACK.md: matrix completeness, matched comparisons,
  clause-level evidence including failed examples, reference/proxy distinction,
  metrics with denominators, caveats, cost and proposed selection rationale.
  Reference each run/observation and baseline member by ID/hash.
- restoration.json: owner checkout/job/Studio/external workload/owned lock and
  process state before/after, no production mutations.
- COMMAND_SHAPE_FRICTION.json: GHDEV-10 canonical fields, all material rejections,
  requested/effective command, provenance, effect, recovery and stop reason;
  zero events explicitly stated if none.
- manifest.json: criterion/path/bytes/sha256/source+runtime identity/privacy/
  limitations and missing artifacts; SHA256SUMS covers manifest and all
  allowlisted payloads, not itself. No circular hashes.

Use private staging directory outside checkout; preserve raw host logs locally,
include only bounded sanitized evidence needed for independent Central review.
Derived bounded source text is necessary private evidence. Exclude full media,
WAVs, weights, venvs, caches, private configs/credentials, backups and huge
logs. Scan entire allowlist, verify manifest-to-payload equality and safe
archive paths, no links/duplicates, regular files only (directory entries
optional), checksum all bytes, chmod 0600. Create archive outside input tree
then verify delivered archive SHA256/size/readability. No GitHub writes.
Return direct conversational status per arm PASS-executed/FAIL/BLOCKED (not
production acceptance), required/included/missing artifacts and reasons,
TESTED_SOURCE_SHA, GITHUB_EVIDENCE_ROOT=none, GITHUB_EVIDENCE_COMMIT=none,
EVIDENCE_PACKET_REQUIRED=yes, EVIDENCE_ARCHIVE_REQUIRED=yes,
PRIVATE_ARCHIVE_ADDS_REQUIRED_EVIDENCE=yes, PRIVATE_ARCHIVE_UPLOAD_REQUIRED=yes,
CENTRAL_REVIEW_ATTACHMENT_REQUIRED=yes, EVIDENCE_PRIVACY_CHECK,
EVIDENCE_INTEGRITY_CHECK, EVIDENCE_ARCHIVE_STATUS, EVIDENCE_ARCHIVE_PATH,
EVIDENCE_ARCHIVE_SHA256, EVIDENCE_ARCHIVE_BYTES, EVIDENCE_MANIFEST=manifest.json.
If unavailable, mark BLOCKED. End with exact required operator action to
attach archive directly to Central. No implementation/deployment work afterward.
