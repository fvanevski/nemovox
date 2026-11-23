# Issue #14: transcript-first architecture decision and evidence gates

Status: **PROPOSED CENTRAL ARCHITECTURE / NO CODE OR MODEL INSTALL AUTHORIZED**
Review date: 2026-11-22 (PST)
Repository: fvanevski/nemovox
Inspected source: 4c32acf659891fe999d2ae6e15f59ccb5750d92f (tree a4e866f53899fe159c99793ff8a78326a10b26f5)
Base main: 8f6c03cbc70e84bf1970ba364207a02c9fe22b44
Current draft PR: #15 (remain draft until independent exact-head gates)
User goal: personal, fully local foreign-language subtitle generation, prioritizing complete, natural source-language transcription and reliable word timing as the foundation for translation, contextual correction, speaker placement, and styling.

## Evidence governing this proposal

- Original independently audited French 22-case baseline: \`5bf6f37d173cfcc160fdf03b60b051cd0f5e7dea:staging/quality-audit/mcog-s04e01/evidence/20261122T205259Z/\`.
- Bounded nine Nemotron retries and ASR-integrity gate: \`11a0ab8bfa0a9be4c67fcc2b822cd529abce5461:staging/issue14/evidence/20261122T231624Z/\`.
- Pause-aware 35-chunk/23-minute French full-run: \`a1812212f6e5c7a82bcd0a27769130fda420a3d3:staging/issue14/evidence/20261122T225739Z/\`.
- Latest independently *host-auditioned*, CPU-only audio-truth and held-out controls: \`36fbf98021094b58efac5de76c7a3432f0a3a7f8:staging/issue14/evidence/20261123T000342Z/\`.
- User's \`transcribe.md\` dated 2026-11-22, independently compared against the primary vendor documentation linked below.

The full episode improved fragmentation (90→40 micro-utterances, 49→12 short shards) but these are NOT source-quality accuracy measures. The latest acoustic review falsified an earlier false-positive "good" W04b/W04c result: a monotonic transcript was ~2s late and omitted ~23 words of narration. It found an H2 ~8-word silent omission undetected by the Nemotron diarizer, 3 theme-song false diar triggers, c28 laughter/SFX false triggers, a four-word hallucination on loud SFX, and Mandarin CTC character fragmentation creating a *false visual overlap*. These are hosted acoustic-listening attestations, not audio personally independently heard by Central; preserve this evidence boundary.

The prior c28 completion gate (finish=length) is a necessary structural guard but cannot establish coverage when a retry returns finish=stop and virtually no words. A VAD's positive speech prediction is also an *observation*, not speech transcription truth.

## Vendor documentation corrections and model selection

1. NVIDIA states Nemotron 3.5 supports *up to one hour* audio and a 65,536-token context, but its encoder's approximately 30-second post-feature chunks are internal to one model forward. Neither feature guarantees accurate one-pass 23-minute transcripts, globally stable speakers, nor accurate CTC timestamps on animated TV. The model card warns that a dense hour may require ~20,000 output tokens; full-input memory/headroom and completion limits need actual validation. Its long-form leaderboard uses 55-minute blocks and 5s overlap and BF16 results, not our NVFP4 model's specific animated-dialogue performance. NIM throughput is measured on a 96 GB RTX PRO 6000 with high concurrency and different optimizations, not single-request 5090 latency. Preserve the host-validated RTX 5090 admission envelope; do not transplant the 96 GB recipe's \`gpu_memory_utilization=0.90, max_num_seqs=32\`.
   - https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4
   - https://github.com/NVIDIA-NeMo/Speech/blob/main/docs/source/speechlm2/models.rst
2. Keep Nemotron 3.5 NVFP4 as the *initial primary recognizer* because it works with the current owned model resource, CTC middleware, French and Mandarin, and has produced material improvements. One-shot full-episode ASR is an optional comparative mode, not the default. Use audio-region-guided bounded inference and batched independent requests; do not assume a 39/46/90-second number is an immutable optimum.
3. Qwen3-ASR-1.7B is a **potential independent challenger** only for contested source regions, not a blanket replacement. Its word timestamps come from pairing with Qwen3-ForcedAligner-0.6B; do not describe standalone Qwen3-ASR transcript text as carrying independently verified word timestamps. The Qwen3 forced aligner supports French and Chinese, up to five-minute samples in official usage, and can be loaded once for **batched per-region alignment**, avoiding previous per-span subprocess-startup overhead. A forced aligner conditions on supplied words: it cannot establish whether those words were really uttered, so it MUST be validated against independent audio anchors and reject unalignable/hallucinated text.
   - https://huggingface.co/docs/transformers/model_doc/qwen3_asr
   - https://github.com/QwenLM/Qwen3-ASR
   - https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py
4. Independently evaluate **Silero VAD** (small, 8/16kHz activity map) and **pyannote Community-1** (offline whole-file regular and exclusive speaker diarization). Community-1 first requires Hugging Face license/condition acceptance and token to acquire the weights, although it operates offline afterwards; do not assume token-free acquisition or silently download model weights. VAD/diar are correlated acoustic classifiers, not voice-versus-lyrics/SFX ground truth. Do not hard-skip detected non-speech or sung lyrics until proven safe on this corpus.
   - https://github.com/snakers4/silero-vad
   - https://huggingface.co/pyannote/speaker-diarization-community-1
5. WhisperX/Whisper + separate aligner are optional baseline/adapters, NOT guaranteed superior for concurrent speech. The WhisperX project explicitly notes limitations on overlap.
   - https://github.com/m-bain/whisperX

## Non-negotiable outcome invariants

**The authoritative content is the original audio and explicit independent annotations, never a previous model response or test's internal status counter.**

- Every acoustically supported spoken occurrence has a preserved, correctly ordered source-text representation or an explicit unresolved/omitted interval; never silently drop words because they are short, grammatical particles, repeated, fall outside one preferred chunk, or resemble another cue.
- A single acoustic utterance should not be rendered twice solely because multiple ASR observations contain it. Genuine repetitions and simultaneous different-speaker speech are distinct source events.
- Source text accuracy and timing accuracy are separate; a monotonic CTC sequence, \`finish=stop\`, forced aligner output, or Nemotron diar span cannot by itself certify either. No fabricated high-confidence word times. Word/character granularity reflects language (French lexemes vs Chinese segmentations/characters), avoiding CJK skip-all rules.
- Speaker IDs are episode-level hypotheses derived from actual source turns and acoustic evidence; request-local \`<spk:N>\` integers must never be globally equated. Preserve regular overlapping turns, and use exclusive diarization only as a *rendering simplification* when defensible.
- Every English translation and rendering cue references one or more immutable accepted source-turn IDs. Translation/ASS renderer never deletes a source event using English text similarity alone; Issue #9 owns layout but must honor this source-event preservation contract.
- Preserve original source language, raw ASR attempts, audio offsets, correction lineage and lexeme choices. Source-text corrections can be context/glossary-assisted without per-edit audio replay when limited to evidence-supported orthography/name consistency; content changes that add/remove speech require stronger acoustic support or remain review-needed.
- No unrelated model-service eviction, modified copyrighted audio, secret/model weights in public repo, or silent downgrade to alternate model/device. No numeric speaker count or fixed overlap threshold as a quality target.

## Proposed minimal data contract

Avoid a generic graph database or elaborate orchestration framework. A few versioned typed records with stable IDs, immutable observation history and explicit accepted revisions suffice:

\`\`\`
AudioRegion { id, media_sha, language, time_window, signal_observations[], coverage_status }
ASRObservation { id, region_refs[], excerpt_start/end, audio_sha, model+params_sha,
                 raw_sha, finish_status, source_text, token_sequence, CTC, local_diar,
                 generated_tags, alignment_status, lexical_coverage_status }
SourceTurn { id, media_sha, source_interval, source_text_revision,
             lexical_units[], speaker_turn_ref?, event_class,
             evidence_obs_ids[], status: accepted|uncertain|omitted }
WordAlignment { source_turn_id, source_text_revision, method, unit, begin/end,
                diagnostic, status: accepted|disputed|unresolved }
SpeakerTurn { id, audio_interval, global_speaker_ref?, overlaps[], evidence[], status }
TranslationRevision { source_turn_ids[], Riva_original, review_candidate,
                      accepted_english, reason, original_hash }
SubtitleCue { source_turn_ids[], start/end, text, speaker_ref?, layout/style }
\`\`\`

IDs are assigned stable persisted identities and explicit revision links, not regenerated solely from mutable CTC timestamps, sequential \`utt-XXXX\`, or speaker tags. Do not duplicate full raw audio in these records; references, offsets and content hashes suffice.

## Proposed processing sequence, optimized for source correctness

1. **Decode/inspect audio once.** Preserve exact media timeline and source track. Build a *soft* activity map from FFmpeg low-energy boundaries plus a separate speech-activity model. Label candidate dialogue, sung speech, other vocalizations, music/SFX and unknown, not just binary silence/speech. For overlaps, preserve both acoustic regions rather than forcing one exclusive assignment. Activity is evidence for review/retry, not authority to delete.
2. **Plan bounded ASR observation windows globally.** Choose inexpensive approximate cuts from source audio activity, target model input duration and overlap penalties, with short boundary padding. Treat core boundaries as scheduling units, *not lexical ownership*. A shortest-path/dynamic-programming cost function can replace \`detected_pause_early_long\` and branch-specific 1s/22s exceptions; benchmark before choosing duration bands. Batch independent requests within owned 5090 resource/latency budget. Allow optional whole-episode inference only as a separately measured challenger.
3. **Acquire ASR candidates, do not immediately commit text.** Nemotron returns immutable observations and checks structural validity. Missing voice activity vs sparse text, contradictory overlap excerpts, \`finish=length\`, CTC jumps across verified acoustic gaps, or unknown speaker turns generate bounded review/retry requests. VAD false positives on singing/SFX do NOT automatically trigger unlimited retries. A fallback ASR model, if installed and licensed, can help adjudicate contested regions; do not require it for every clip.
4. **Reconstruct the canonical source transcript before translation.** Group overlapping observations around source acoustic regions using monotonic lexical correspondence and *independently checked* coarse timing. Do not globally trust two matches because they were both produced by Nemotron. Prefer preserving uncertain source text and an explicit review state over deleting or fabricating an occurrence. Genuine repeated calls across distinct acoustic events survive. Version all accepted source-turn revisions.
5. **Run source-language correction, then independent word alignment.** Correct accents, names, spacing and punctuation against cross-turn context and optional known glossary; retain original transcription and mark substantive wording edits as evidence-limited. Perform short/turn-level batch forced alignment of accepted source text to *its corresponding source audio*, preferably with Qwen3-ForcedAligner-0.6B if host-quality tests warrant it, and cross-check Nemotron CTC. Reject alignments that cross independent pauses, collapse on SFX, omit expected units or cause impossible time order. Do not trust output merely because all timestamps are finite or monotonically increasing. Use French words and suitable Mandarin word/character units; make character-level fallback conditional and batched rather than launching per-char subprocesses.
6. **Assign consistent speaker turns globally.** Compare full-media pyannote Community-1 regular diarization with existing Nemotron/TitaNet evidence on labeled French/Mandarin turns. Use independent acoustic speaker boundaries; infer linkages only where support is adequate and preserve unknowns. Align accepted source turns to speakers after source/timing reconciliation (or jointly if actual overlap demands it). Do not force the same speaker identity because of lexical similarity or fixed target speaker count.
7. **Translate and render last.** Riva translates source-verified coherent turns, Qwen reviews English with source/neighboring turn context and original Riva visible; retain helpful corrections and reject unsupported meaning shifts. Renderer consumes the source turn IDs and real timing; it may split/merge display cues but MUST NOT remove source events merely to achieve low CPS or text similarity. Spatial placement needs visually verified speaker/scene evidence when audio cannot locate the speaker; style classes may distinguish speech, song and effect captions without altering transcription.

## What to remove or replace, not patch around

- Retire production token-by-token boundary deletion, extended right rescue, and \`SequenceMatcher\` hard-length caps **after** event-correspondence replacement passes the acoustic corpus. The new method owns duplication and omission decisions; no parallel fallback owners.
- Replace the isolated CTC flanked-outlier heuristic with stage-separated structural checks and acoustic/forced-alignment validation; keep raw CTC available as evidence.
- Replace arbitrary single TitaNet cosine cutoff and acoustic-guessing speaker ID counts with labeled pair/turn validation and episode-level diarization.
- Remove downstream English clause similarity dedup and blind Mandarin single-character cleanup where it deletes accepted source events. Issue #14 defines source preservation; coordinate exact code ownership with Issue #9.
- Retire tests that require old chunk winners, sequence_promoted counters, exact 64.241427s cut or \`utt-XXXX\` ordinal IDs. Preserve *observable* clean/repetition/overlap/source-coverage invariants. Keep fixed-policy backwards compatibility as a separate API/cache migration test, not the preferred long-term audio strategy.

## One integrated validation gate, not a chain of campaigns

Before applying new VAD/aligner as authority, compare existing locally available models (no forced download, no new paid service) against the already private **independently heard** 18 French regions, 2 Mandarin windows and held-out H1-H4. Run a single small host-controlled qualification *as part of the Central-authored replacement integration*:

- **Speech activity:** Silero VAD vs optionally installed pyannote segmentation, on H2 omitted speech, c28 true shouts vs screams/laughter, W04c/female narration, song/SFX and clean controls. Report event-level detection, false alarms/omissions, not an invented whole-episode sensitivity.
- **Independent word timing:** Resident batched Qwen3-ForcedAligner on short, source-audited French/Chinese transcripts vs Nemotron CTC. Compare to independently labeled time bands (median/tail errors, rate of aligned-but-false text, unknowns) and check a deliberately false W04c transcript candidate is not accepted because of monotonic times. If model is unavailable, record \`UNVERIFIED\` and keep CTC as provisional; no silent model download.
- **Global speakers:** One whole-episode pyannote pass only if already locally accessible within approved resource lifecycle; compare true overlapping/single-speaker turns at W02b, S05 and Mandarin 525s. The full-episode run is a diarization evaluation, NOT another full Nemotron ASR run.
- **Optional challenger:** Qwen3-ASR-1.7B on only the missed/sparse/singing/SFX and clean control excerpts if locally available and model resource identity is authorized, to test whether it adds previously missing actual words with acceptable hallucination rates. Do not interpret simple ASR-to-ASR agreement as audio truth.
- **Performance:** number of loaded models, real GPU peak, batch throughput, extra aligner cost, number of retries, time-to-first-source transcript. Prefer sequential load/release on single RTX 5090 over concurrent GPU contention. Do not repeat full 23-minute ASR just to score model features.

Then Central authors a single cohesive replacement for region/event reconstruction, source-text correction/word alignment and speaker mapping on **existing PR #15**. Use portable CPU regression tests from source-event outcomes, and one final genuinely independent host full episode/held-out French+Mandarin check before Codex independent review and operator merge decision. The new implementation is qualified only if it reduces verified omission/duplication and timing/speaker errors, with runtime/VRAM trade-offs explicit. Do not count lower utterance/ASS totals as fidelity. No further one-heuristic micro-slices.

## Unresolved choices requiring evidence or operator architecture approval

1. Which locally available independent VAD/aligner/diarization model provides measurable improvement on the user's TV/anime corpus, and what it costs on one GPU. **Unverified until host-tested.**
2. Whether dual ASR candidates are useful for low-coverage regions or add too much latency. **Unverified.**
3. Whether the source-turn representation and rendering event preservation interface should replace old contract v1 in-place or use an explicit versioned migration. Favor versioned output with compatibility adapter, do not silently break previous jobs. **Operator architecture scope decision before irreversible migration.**
4. Actual episode-wide acoustic truth and speaker count remain UNVERIFIED; no claim based on 28 model cluster IDs.

## Relevant primary technical sources (read on 2026-11-22)

- NVIDIA NVFP4 model card: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4
- NVIDIA NeMo Speech long audio / encoder chunks: https://github.com/NVIDIA-NeMo/Speech/blob/main/docs/source/speechlm2/models.rst
- Qwen3 ASR and ForcedAligner API/11 languages: https://huggingface.co/docs/transformers/model_doc/qwen3_asr
- Qwen3 ASR implementation (batch alignment and low-energy splitting): https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py
- Pyannote Community-1 official model card: https://huggingface.co/pyannote/speaker-diarization-community-1
- Silero VAD: https://github.com/snakers4/silero-vad
- WhisperX limitations: https://github.com/m-bain/whisperX
