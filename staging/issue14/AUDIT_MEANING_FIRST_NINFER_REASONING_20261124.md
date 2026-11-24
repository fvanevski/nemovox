# Issue #14: Tiered meaning-first QC host failure and Central correction

## Governing objective

Natural, grammatical, readable English subtitles that convey contextual intended meaning, polarity, causal relationships, and essential speaker/scene continuity, with useful event-level timing. Not literal per-word source reproduction, complete audio-reference WER, guaranteed recovery of background fragments, or perfect word timestamps. A fluent output is not proof of a source-acoustic occurrence: immutable source-unit identities, alternate ASR hypotheses and actual reviewed-audio evidence remain distinct.

## Independent handback audit

Host handback ISSUE14_MEANING_FIRST_QC=PARTIAL at evidence-only PR head e757c3161c7dac1e5a2228aa44774f30722586c0 and validated source/test 622715afc00dd6d215fe27c9b5844f615d762971, tree 1317b67868baec61852e9d84d087b0e0e3994d5d. Uploaded issue14-meaning-first-qc-20261124T013230Z.tar.gz independently reopened: 33195 bytes SHA256 1e4e932d76fd9c8ce6329a59a9f72a7ee? CORRECT FULL: 1e4e932d76fd9c8ce6329a104b7e23b468e13b5a915a7bc224; seven unique regular tar members; six unique CHECKSUMS.sha256 entries, all matched. Previous forensic archive 52568c845ab3e7c2541e3711b8950cccde93a11164cb7eadf3c706d4bf923b8e host reauthenticated 21/21. Current public packet staging/issue14/evidence/20261124T013230Z has eight manifested files and manifest itself, all eight remote files independently read with Git blob IDs and matched byte counts; public SHA256 member checks remain host-attested.

CPU host gates attested: focused 66/66, core 453/453, Studio 453/453, compileall and diff --check clean; exact candidate imports. This verifies PREVIOUS source, not the new correction.

T0 real replay: Run A 3 candidates, Run B 6, no automatic source rewrite, original merged units unchanged (A 2008; B 2002). High-importance S02 350-359.8s including negation and W01c 80.2-83.254s boundary context detected. C1, C2, W04c SFX and Mandarin character control no false promotions; W04a one advisory-only candidate. Interleaved source-speaker words capped S02 lexical correspondence at four when global left stream was used. This is review-evidence fragmentation, not proof of missing acoustic source.

## Root-cause: model request capacity, not a failure of reasoned adjudication

All six owned NInfer/Qwen3.8 model requests used reasoning_effort high for complex cases and low for one simple case. First high value generated HTTP 400 "invalid prompt | messages 2"; OpenAIChatTextClient originally set one sticky _reasoning_rejected flag, silently discarding the requested effort for every later case. The server retained its --thinking-budget 16384 default. All five high-tier 1280-token and one low-tier 512-token responses had finish_reason=length, empty final content and reasoning_tokens equal completion_tokens. Six unaccepted JSON assessments; model did not reach a final decision. These are genuine request/server incompatibilities, not evidence that low- versus high-reasoning QC is unhelpful.

Upstream corroboration: https://github.com/Neroued/ninfer/issues/440 documents older Qwen3.8 chat templates accepting low/medium/xhigh but rejecting high. https://github.com/Neroued/ninfer/blob/master/docs/serving.md documents per-request thinking_budget in Chat Completions and that the budget cap and forced-close overhead count against max_tokens. This is provider/version-dependent; current host executable behavior still requires revalidation, not assumptions based on upstream master.

## Central staged correction

Latest SOURCE/TEST commit 4c27c5b8d13a173aa8ce2f3c1f9699a37d8a14c0, tree 7fb880ff1614f4ce21ab28a87fda5de3e62c62c0, subsequent CONFIG-only commit a31a7d7d25b083671126f9f63b56151a7526a0d2. All source changes on existing PR #15.

- src/nvidia_transcribe/refinement.py, shared OpenAIChatTextClient: optional per-request NInfer thinking_budget, strictly reserve at least 128 final-answer tokens; explicit strict_reasoning_effort mode refuses unsupported effort rather than falling back to unbounded default. Legacy tolerant mode remembers unsupported values individually, not via sticky global denial; low still works when a separate high value is rejected. Response includes sanitized request-control diagnostics.
- src/nvidia_transcribe/semantic_review.py: Run T1 low and T2 xhigh by default under explicit different per-case thinking caps (128 and 512) and total completion ceilings (512 and 1536 default; example T2 1280), rather than the 16384-token server default. Report reasoning vs completion use, finish reason and empty-content budget failures; no unaccepted response becomes a correction. Versioned run_configured_tiered_text_reviews verifies the same policy digest used by candidate planning.
- T0 review evidence now compares each left local-voice stream separately before selecting a best same-words sequence match, retains ties and competing voices as hypotheses, and does not infer global voice identity from local labels. No source transcription changes or automatic duplicate removal.
- tests/test_issue14_ninfer_thinking_budget.py + tests/test_issue14_semantic_review.py: synthetic transport response for the actual invalid-prompt case, xhigh/low tier survival, strict non-fallback, final-answer cap reservation, reasoning-only response fails closed, interleaved voices, version-bound execution config, and no source mutation.
- configs/meaning-first-review.example.json includes tunable model_review (6 case ceiling, total max 6912 completion tokens, low 512/128 and reasoned 1280/512 total/thinking, low/xhigh effort). It is an explicit opt-in host-controlled text-model profile, not a server-global override.

Independent developer-built local model reasoning was NOT replayed after these changes; host suite and actual NInfer requests remain UNVERIFIED. Do not claim positive LLM adjudication until the new host handback demonstrates structured, evidence-appropriate decisions with actual final content and clean ModelLifecycle restoration.

## Remaining boundary

This stage restores valid T1/T2 responses and better cross-chunk evidence packaging. It does not automatically implement S02/W01c output corrections or replace the existing reviewed-source acceptance gate. A meaningful English paraphrase grounded in a reviewed turn may be accepted by translation post-edit; new acoustic/source events require positive source evidence. After one consolidated candidate-bound host campaign, Central should independently assess whether one generic advisory-to-reviewed-correction bridge is necessary to meet Issue #14, then perform exact-head PR review. No new full 23-minute inference campaign or merge/issue closure is authorized.
