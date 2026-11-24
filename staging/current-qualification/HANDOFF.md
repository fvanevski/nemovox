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
  --output-dir "$new/live" --model-resources "$resources"
```

If current resource selection is through NEMOVOX_MODEL_RESOURCES, omit the explicit flag rather than invent a path. Script extracts four fresh regions (French omission, two Mandarin gap portions, ordinary French control), compares speaker/verbatim/multispeaker_sot on identical audio, and obtains two direct-English diagnostic responses. 14 ASR calls max, one owned ASR scope. Direct English is a diagnostic and cannot replace source transcript/MT or supply timing. Record raw generation, CTC, original-relative timing, prompt mode, actual model/quantization and request receipts. Same checkpoint and different prompt do not constitute independent-model evidence.

After ASR releases, the script performs 8 bounded NInfer Messages calls in one text scope: four source-first interpretations (no old English shown), then blind comparison against primary MT and previous reviewed proposal. It uses existing exact source/context from verified prior contracts as diagnostic inputs, not acceptance data. Verify actual 2048 thinking budget and 4096 completion cap; no fallback or retries to weaker transport. Review gloss versus actual English, not just choice or evaluator self-confidence. Explicitly assess French idiom and restrictive French negation vs genuine negation, Mandarin economic action vs dispute resolution, and the two previously helpful controls. Do not give model desired answers in prompts. Source-first selection is not automatically correct.

**Independent acoustic comparator, if available without installing or downloading anything:** inspect existing faster-whisper environment and cached CTranslate2 checkpoint (previous host work may have one, current availability unverified). Run the staged optional CPU-only diagnostic using its actual existing Python and explicit local checkpoint, on the very same four WAVs. This is a diagnostic, not production CPU fallback or an automatic model substitution.

```sh
rtk proxy env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$whisper_python" \
  staging/current-qualification/cached_asr_probe.py \
  --checkpoint "$cached_ct2_model" --inputs "$new/live/inputs.json" \
  --output "$new/live/independent-asr.json"
```

Before constructing the model, confirm local model.bin, config and tokenizer/preprocessor assets in that checkpoint or existing cache; do not rely on environment flags alone. Do not modify /workspace/pyannote or its venv. If no usable local independent checkpoint/runtime exists, record exact unavailability (do not install, switch to GPU or broaden search). If available, compare its actual source speech and event times with Nemotron results; do not compare WER against unverified historical text. A brief operator listening judgment on actual excerpt is useful, especially for French; report if exact language content is not independently confirmed. Independent model agreement is evidence, not ground truth.

## Handback and next decision

One consolidated handback. No full-run continuation from this diagnostic. Return separately:
- raw generation/CTC/reconstruction/selection cause for each consequential gap, with actual private raw/CTC observations and absolute source bounds;
- which prompt or independent cached ASR materially improves speech coverage, timing or meaning, which remains unreliable, whether a short independent aligner is justified; include positive normal controls and timings/VRAM;
- source-first and blind text results for the four cases, semantic comparison and material regressions, no claims based only on self-evaluation;
- native commands/exits, script version/head/tree and any mechanical diff, resource identity, owned units/ports/lock restoration, hashes and manifest; exact distinction between host evidence and model claims;
- COMMAND_SHAPE_FRICTION in canonical GHDEV-10 form.

Keep raw source text, excerpts, full model responses and host paths private. Publish sanitized summary and manifest/checksums only on existing branch under staging/current-qualification/evidence/<UTC>. Return private review-ready tar.gz with actual bounded input/output details and sha256; no original full media, weights, secrets or large unrelated files. Attach private archive for Central. Stop once decisive comparison complete; do not improvise production fixes or new model campaigns.
