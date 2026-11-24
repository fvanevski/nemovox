# Current-workflow quality investigation

Status: **bounded live quality FAIL at application c4f339288a984e0517ea1cb94b579cdc0d2c3597**. Evidence-only carrier b4f56e84fff59157546118f445a86d4d04839d4e. Previous CPU, short-overlap, media-end, one-command CLI, Studio execution, offline rerender and lifecycle results remain valid only for their exercised application/workload. Do not repeat them merely to increase test counts. Full episode remains gated out. No production implementation is changed in this diagnostic continuation.

## Purpose and limits

One decisive host investigation before further Central implementation. The latest evidence shows near-identical event bounds but different acoustic wording in French, a long internally shared Mandarin sequence with unreliable edge tokens, and a text reviewer that endorses an English action inconsistent with its own gloss. All acoustic challenges currently use the same speaker preset. Do not tune thresholds, raise budgets, patch production or invent episode-specific fixes. Do not qualify by matching an old transcript. Distinguish missing raw generation, CTC-unit/reconstruction loss, selection loss, and intrinsically uncertain lexical meaning. Central will adjudicate the evidence and implement the next production correction. This is not a broad model benchmark or mandatory new checkpoint.

Use the exact head/tree in the kickoff, read AGENTS.md, verify imports in an isolated worktree and preserve operator work. Build role is authorized only for host scratch/evidence writes, branch publication of sanitized reports and narrow diagnostic-script mechanical corrections (full diff, exact head). No product changes, new PR, ready transition, merge, deployment, issue closure, model download, dependency changes or unowned service operations.

## CPU-first, no GPU acquisition

Use the previous private host root `/workspace/nvidia-transcribe/outputs/qualification-14-remediation-VEyCYnAj` and the checksummed `private/raw-evidence-manifest.json` from the previous archive (or its verified original host-local copy). Verify previous archive digest 91bfdc21cd6ea35d7024e41ebc98e47781512a2b69ad41569d584f79eb6bfec6 and all used input hashes. If the manifest is not materialized, safely extract that exact member into a new private scratch directory; no overwrite or extraction of arbitrary archive paths. Use the original media paths and hashes in the previous private `original-media.sha256` file; verify they match before any live work.

Run `python -m py_compile staging/current-qualification/probe_quality.py staging/current-qualification/cached_asr_probe.py` with the existing project .venv and confirm import location. Use rtk proxy where required, native exits and logs. Run the following with actual private paths, new output root outside prior job directory:

```sh
rtk proxy env PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python \
  staging/current-qualification/probe_quality.py \
  --prior-root "$prior" --prior-manifest "$manifest" \
  --output-dir "$new/inspection" --inspect-only
```

Inspect original raw generated text, CTC payload and reconstructed original rows vs canonical source and prior acoustic observations. For French absolute 790.56–795.15 s, determine if the *original* raw ASR text contains the omitted speech, whether the raw CTC payload loses it, whether boundary ownership or reconstruction dropped it, or whether it was absent until probes. For Mandarin original 238.257–257.880 s, trace both untimed rows and any duplicate fragments, actual unit timing and CTC error, and distinguish prior reconstructed input from raw original. Record shared internal sequence anchors, actual unit spans, which original row they plausibly identify, and any competing row, without asserting timestamps for unmatched edges. Include actual private CTC/unit evidence from existing prior host files. No model calls for this step.

## One bounded live capability comparison

Only after CPU inspection and resource admission, use the same existing model configuration and local checkpoints. Verify original media digests, actual selected stream 0, owned service/lock state, VRAM and current versions. No full episode, no new Studio job, no repeat of the three prior bounded jobs. Run:

```sh
rtk proxy env PYTHONPATH=src:. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /workspace/nvidia-transcribe/.venv/bin/python \
  staging/current-qualification/probe_quality.py \
  --prior-root "$prior" --prior-manifest "$manifest" \
  --fr-media "$fr" --zh-media "$zh" \
  --output-dir "$new/live" --collect-only --model-resources "$resources"
```

If current resource selection is through NEMOVOX_MODEL_RESOURCES, omit the explicit flag rather than invent a path. The script extracts four fresh base regions (French omission, two Mandarin gap portions, ordinary French control), compares speaker/verbatim/multispeaker_sot on identical audio, and obtains two direct-English diagnostic responses. For each problematic target it additionally extracts an expanded window (10 s padding per side, clipped to media) and two shifted-boundary windows, with a nominal midpoint boundary and 6 s inference overlap on each side. The geometry comparison uses speaker mode only, holding the prompt fixed. Record actual crop/core boundaries and compare target coverage, lexical differences, reliable timing, and duplicate/turn risks across the overlapping halves. Reconcile observations diagnostically; do not replace the production reconciler or treat overlap as proof of duplicate speech. 20 ASR calls max, one owned ASR scope. The script also extracts five private screenshots per problem region (before, within, after); preserve hashes, seek times and decoder logs. Image extraction failures are explicit, not a reason to abort the acoustic comparison. Direct English is a diagnostic and cannot replace source transcript/MT or supply timing. Record raw generation, CTC, original-relative timing, prompt mode, actual model/quantization and request receipts. Same checkpoint and different prompt do not constitute independent-model evidence.

Collect independent audio and visual evidence before the separate text assessment below. The collection-only phase performs no contextual text calls.

**Independent acoustic comparator, preferred whenever already available:** inspect existing faster-whisper environment and cached multilingual Whisper large-v3 (or competent multilingual large/turbo) CTranslate2 checkpoint; prefer strongest already provisioned multilingual model. Previous host work may have one, current availability unverified. Run the staged optional CPU-only diagnostic using its actual existing Python and explicit local checkpoint, on the very same four base WAVs and two expanded problem WAVs by default. This is a diagnostic, not production CPU fallback or an automatic model substitution.

```sh
rtk proxy env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$whisper_python" \
  staging/current-qualification/cached_asr_probe.py \
  --checkpoint "$cached_ct2_model" --inputs "$new/live/inputs.json" \
  --output "$new/live/independent-asr.json"
```

Before constructing the model, confirm local model.bin, config and tokenizer/preprocessor assets in that checkpoint or existing cache; do not rely on environment flags alone. Do not modify /workspace/pyannote or its venv. If no usable local independent checkpoint/runtime exists, record exact unavailability (do not install, switch to GPU or broaden search). If available, compare its actual source speech and event times with Nemotron results; do not compare WER against unverified historical text. A brief operator listening judgment on actual excerpt is useful, especially for French; report if exact language content is not independently confirmed. Independent model agreement is evidence, not ground truth.

## Visual context and multimodal adjudication

The screenshot manifest is in live/inputs.json. Inspect the frames themselves, not filenames. Record whether the sequence actually helps distinguish speaker turns, actions/objects, narrative referents, visible scene changes or existing onscreen writing. Screenshots can inform meaning, not prove verbatim speech or exact speaker identity. If a genuine image-capable local model is already available with a verified, permitted image-input route, use it for one bounded visual description per problem scene (five frames in chronological order); preserve model/checkpoint identity, actual image request/response, and the frame IDs/hashes. Do not supply proposed acoustic wording to the initial visual inspector. Do not route images through the text-only NInfer Messages adapter, which explicitly accepts only string content. No model downloads, new adapters, remote media upload or unowned GPU operations. Use the existing permitted local multimodal harness/resource only after ASR releases. If unavailable, return all screenshots in the private archive for Central visual inspection; do not claim local VLM execution. An actual operator observation is admissible if explicitly distinguished from model description.

For local model or operator descriptions, prepare private JSON with this shape (one entry per actually described frame; no placeholders or invented observations):

```json
{"observer_identity":"actual model/checkpoint or operator",
 "frames":[{"frame_id":"actual manifest ID","sha256":"actual image digest",
            "description":"visible evidence, not inferred dialogue",
            "uncertainty":"what cannot be determined"}]}
```

For each problem, preserve direct visible observations separately from interpretations; a scene sequence can be useful even when no visible speaker is identified. If visual evidence is unavailable, omit the optional description argument, not the screenshots.

## Contextual reasoning and combined evidence assessment

After all admitted acoustic/visual collection and model release, execute the separately staged text phase (single owned NInfer Messages scope):

```sh
rtk proxy env PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python \
  staging/current-qualification/probe_quality.py \
  --prior-root "$prior" --prior-manifest "$manifest" \
  --output-dir "$new/assessment" --assess-collected "$new/live" \
  --model-resources "$resources"
```

When available, add --independent-asr "$new/live/independent-asr.json" and/or --visual-observations "$new/live/visual-observations.json". Omit unavailable inputs; do not fabricate JSON or require a second text acquisition merely for them.

For each problem the first contextual assessment sees ONLY prior acoustic hypotheses and original surrounding source rows, not newly collected data, old reviewer rationales, or the expected answer. The second, separate reasoning call sees original context, prior and new acoustic observations, independent ASR if available, and actual screenshot descriptions if available. It does not see the first verdict. Assess whether the initial contextual cues suffice to resolve intended meaning, which uncertainty requires more evidence, and which later evidence actually changes the result. This evidence-isolated diagnostic comparison is intentionally run after collection to avoid extra GPU reloads; it measures a possible early exit, not actual avoided inference. In production the contextual decision should precede additional acoustic collection when existing evidence is adequate. Do not treat meaning resolution as timing verification.

The text phase also runs the original four source-first interpretations and four blind MT comparisons. Total 12 Messages calls (4 contextual/fusion + 8 translation), thinking budget 2048 and max_tokens 4096. Verify receipts and assess actual semantic fidelity rather than evaluator confidence. No transport fallback. If a vision-capable model could not run locally, Central can inspect returned frames before deciding the production visual route; mark multimodal reasoning partial.

## Handback and next decision

One consolidated handback. No full-run continuation from this diagnostic. Return separately:
- raw generation/CTC/reconstruction/selection cause for each consequential gap, with actual private raw/CTC observations and absolute source bounds;
- which prompt, expanded context, shifted overlap or independent cached ASR materially improves speech coverage, timing or meaning; include paired crop/core comparisons, shared observations and remaining competing events, positive normal controls and timings/VRAM;
- context-only versus combined-evidence decisions: cite actual alternatives and contextual discriminators, whether the early decision would safely avoid escalation, which additional observation changes the result, whether timing remains open; include private screenshots, hashes, seek times and actual vision/description evidence or explicit capability blocker;
- source-first and blind text results for the four cases, semantic comparison and material regressions, no claims based only on self-evaluation;
- native commands/exits, script version/head/tree and any mechanical diff, resource identity, owned units/ports/lock restoration, hashes and manifest; exact distinction between host evidence and model claims;
- COMMAND_SHAPE_FRICTION in canonical GHDEV-10 form.

Keep raw source text, excerpts, full model responses and host paths private. Publish sanitized summary and manifest/checksums only on existing branch under staging/current-qualification/evidence/<UTC>. Return private review-ready tar.gz with actual bounded input/output details and sha256; no original full media, weights, secrets or large unrelated files. Attach private archive for Central. Stop once decisive comparison complete; do not improvise production fixes or new model campaigns.
