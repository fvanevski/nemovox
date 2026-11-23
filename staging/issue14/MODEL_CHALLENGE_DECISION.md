# Issue #14: model evidence needed before promoting source-event architecture

Status: Central model-selection plan; no model has been selected as new source, timing or diarization authority.

## Result from the just-completed campaign

Host-reported CPU/Studio and replay tests passed on source 84713a13fd521b3d99b91f09863aff95153c8fff. Central independently checked all seven published evidence members at 1bbcb3c8511e2bf2dfd5798d88d44789068549b5 and verified every listed SHA-256/byte length. Observational replay retains 2,415 old French units (orig16), 2,263 complete 35-window units with c28 rejected, and 3,200 Mandarin units with c14 rejected. These are *different observation denominators*, not measured correct transcript word counts. Some 30+ lexical matches are spurious (e.g. SFX-to-dialogue shifted by ~15.36s). Source-event acceptance, word-timing validation and global speaker assignment are still unproven.

## Compact decision-driven challenger matrix

The model trial is local, isolated, and can fetch missing weights under /workspace/models only; it is NOT an implicit change in the production resource lifecycle.

| Role | Baseline/control | Challenger | What could justify adoption? |
| --- | --- | --- | --- |
| Voice/singing/music activity | existing Silero VAD v6 and pyannote Community-1 | **FireRedTeam/FireRedVAD** (Apache-2.0) | Detect H2, c28, W04c and real song without SFX over-trigger; distinguish speech/singing/music as *soft evidence* |
| Global diarization (episode-level continuity, true overlap) | pyannote Community-1 regular/exclusive, existing Nemotron/TitaNet | **nvidia/Nemotron-3-Diarization** (OpenMDW-1.1) | Fewer false merges/splits on W02b/S05/Mandarin; maintain identity across 23m under real speaker annotation |
| French independent ASR | Nemotron-3.5 NVFP4 raw observations; locally cached Qwen3-ASR-1.7B | **nvidia/parakeet-tdt-0.6b-v3** (CC-BY-4.0) | Better independent recovery of missed French words without hallucinating song/SFX |
| Mandarin independent ASR | Nemotron-3.5 NVFP4; locally cached Qwen3-ASR-1.7B | **FireRedTeam/FireRedASR2-AED** (Apache-2.0) | Better CER and recovery on EP166 missed/overlapped speech without excess inventions |
| Word/character timing | provisional Nemotron CTC | locally cached **Qwen3-ForcedAligner-0.6B** (Apache-2.0) | Lower independent acoustic anchor timing errors for genuine correct source text; NOT proof a supplied transcript was uttered |

Important source corrections:
- pyannote Community-1 is **CC-BY-4.0 but gated by access conditions**. It is already cached locally, so no new acquisition is needed unless the authenticated cache proves incomplete.
- NVIDIA multitalker-parakeet-streaming-0.6b-v1 **consumes external diarization supervision**, needs a model instance per speaker and carries **NVIDIA Open Model License**, not CC-BY-4.0. Its French/Mandarin ability is NOT established in the primary card; exclude it from this language-specific trial. Do not assert independently validated word timestamps.
- Nemotron-3.5 Transcribe already uses a Nemotron-3-Diarization speaker encoder; standalone Nemotron-3-Diarization is a useful additional diarization *execution mode*, not an independent ground-truth labeler.
- Parakeet-TDT-v3 explicitly supports French and **not Mandarin**; FireRedASR2 specializes in Mandarin/English.
- Do not replace the current recognizer based on vendor leaderboards obtained on different distributions. Do not use download counts as accuracy evidence.

Primary documentation (current during planning, refresh exact revision/terms before download):
https://huggingface.co/FireRedTeam/FireRedVAD
https://huggingface.co/nvidia/Nemotron-3-Diarization
https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
https://huggingface.co/FireRedTeam/FireRedASR2-AED
https://github.com/QwenLM/Qwen3-ASR
https://huggingface.co/pyannote/speaker-diarization-community-1
https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1

## Fixed evaluation corpus (no cherry picking)

Use existing independently auditioned 16 French regions and 2 Mandarin windows in staging/issue14/evidence/20261123T004035Z/validation/case-adjudication.jsonl (acoustic truth itself originated from the earlier independently host-auditioned 20261123T000342Z packet). Reuse C1/C2 repeats, W02b genuine alternate voices, W03a duplicates, c28 scream/SFX vs actual word events, W04c + female narration, H2 false-negative VAD, W05a/S05 tail, CTRL song and SFX, and Mandarin 525.16 turn. Source-correct French/Mandarin text must be independently transcribed/auditioned in private before any WER/CER: published approximate word counts are NOT ground truth. Add at most four predeclared independent held-out windows from that same source-identity packet if previous controls are too defect-enriched; no post-hoc relabeling or interval selection.

Model output adapter contract:
- Reference JSON uses `nemovox-issue14-acoustic-reference/v1` with media fingerprint, `regions [{id,start_s,end_s,acoustic_truth,events:[{start_s,end_s,kind}]}]`, all timestamps media-absolute.
- Prediction JSON uses `nemovox-issue14-model-predictions/v1` with same media fingerprint, exact model `{repo_id,revision}`, `regions:[{id,status,observations}]`; explicit status is `ok|not_run|blocked|failed|skipped_unsupported`; every reference region must appear even when unsupported.
- Classes: `speech|singing|lexical_shout|music|sfx|silence|nonlexical_vocalization|unknown`. Preserve voice/singing/mixed labels; do not force all audio into speech or silence. Include speaker-tag outputs separately as opaque evidence, not accepted episode IDs.
- Use `python evaluations/issue14/model_challenge/score.py --reference <PRIVATE_REFERENCE_JSON> --prediction <PRIVATE_MODEL_PREDICTION_JSON> --output <NEW_PRIVATE_RESULT_JSON>`. The scorer's event-level hit is **any temporal overlap**, a coarse proxy; it explicitly does NOT calculate source lexical accuracy, word timing, global speaker DER or product acceptance. Source text, full transcription, and waveforms stay private.
- Separate a source-faithfulness case table with human-auditioned words where available (WER for French, CER for Mandarin), explicit ASR omissions/hallucinations, word-time-band error on an independent sample, short-turn speaker separation/false merge, model load/VRAM/runtime and failing/unknown cases. Do not elevate aligner-monotonic times or model-to-model agreement into acoustic truth.

## Minimal operating contract

First inspect installed models and snapshot identities. Existing Qwen3-ASR/ForcedAligner and pyannote Community-1 are reuse candidates. For missing challenger weights, user explicitly authorizes local acquisition and storage **under /workspace/models**, with original repo/revision/license and digest provenance recorded. Use `hf download <repo-id> --revision <full-HF-commit> --local-dir /workspace/models/<stable-functional-name>` after resolving the current repository SHA and available disk; if the host uses another official HF CLI form, inspect `hf download --help` and adapt. Do NOT download into repository, overwrite existing checkpoints, or modify a production model path. New test-only environments outside /workspace/pyannote or global project VENV are permitted, but use pinned compatible dependencies and no global installs. No user credentials in artifacts. An access-restricted model is skipped if prerequisites are not already accepted; do not bypass a gate.

Model trials must be short, isolated, sequential on one RTX 5090. Do not stop unrelated GPU services. Use the existing Nemovox-owned admission path for model-backed GPU operations; if not compatible with a standalone challenger, authenticate an ordinary short offline GPU task with verifiable GPU headroom and no service changes, or mark blocked. Evaluate all predeclared clips before any optional whole-episode diarization. Whole-episode diarization may be run once for Community-1 and once for Nemotron-3-Diarization **only if** pilot short-clip evidence shows a plausible improvement and the host ownership condition permits it. No 23-minute ASR rerun.

Next Central source work will replace legacy token-by-token boundary deletion with audio/source-event correspondence, bounded speech-recovery requests and versioned accepted source turns, guided by these *measured* models. Do not preemptively make any model authoritative.
