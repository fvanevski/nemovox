# Final Integrated Subtitle Workflow Qualification — Sanitized Public Evidence

- **Run ID:** `20261125-3e6f8d8`
- **Repository:** `fvanevski/nemovox` (Issue #19, Parent Issue #16, PR #21, Branch `issue/19`)
- **Result:** `PARTIAL`
- **Base Commit / Tree:** `4dd4599e51775ca510002783ea650ecd25fd7393` / `dfbc53906300996373c34631fe33029b87bbf313`
- **Prior Evidence Carrier:** `8a304da444a65e688bc7d2ded8f30e8f35924fe7`
- **Incoming Candidate Commit / Tree:** `d4f1d8b9c5baecb90f7f46b75d1c387325a17ca9` / `7a6125a4f3ad30a102d7cc5cd0b93bc33d68e44a`
- **Tested Source Commit / Tree:** `3e6f8d8244f818c304bf5d692e4db864a6ae9a1e` / `3b28fbabf6ec5030b0f1b7752e6b426517f44786`
  - Includes 1 authorized mechanical test-mock repair (`tests/test_translation_review.py`, see `mechanical-test-repair.diff`) supplying the newly required `passage_checks` verdict in `test_messages_source_first_and_comparator_receive_coherent_cross_turn_passage`; zero production code modified, zero assertions weakened.
- **Private Review Archive:** `/workspace/nvidia-transcribe/outputs/handbacks/final-integration-20261125-d4f1d8b/final-integration-review-3e6f8d8.tar.gz` (`21,867,540` bytes, SHA-256 `49e31f742cae801672de81ea604144769d1e7e8f3114b6b4f5f83d07bcb2e1df`)

## Sanitized Summary by Phase

| Phase | Scope | Outcome | Key Metrics |
|---|---|---|---|
| **A** | Deterministic CPU / Studio / Frontend | **PASS** (after mechanical test-mock repair) | `compileall` 0, `diff --check` 0, Focused 120/120, Full CPU 546/546, Studio 51/51, Frontend 14/14 + Vite build PASS |
| **B** | Bounded French Video (110s, exact C2 media & recording) | **PASS** | 329.358s wall; 26/26 context in 1 call; 2/2 reference timing support; joined westward-course passage grammatical across cue boundaries; 6 review calls (63.074s); 1 genuine scene error properly withheld checkpoint |
| **B (Controls)** | Live French, Mandarin & Known-Pixel Controls | **PASS (with honest residuals)** | 53.867s wall; French negation/restriction/split-direction & Mandarin controls passed; disambiguating idiom control retained literal-indifference phrasing (residual); known-pixel title card rejected due to model uncertainty text (Invariant G verified) |
| **C** | Current Resume & Offline Rerender | **PASS** | Attempt 1 legitimately retried after Phase B scene error and created checkpoint; Attempt 2 verified `source_quality_resume=reused_exact_input`, 26/26 Riva & 26/26 review cache hits, 0 API calls, 0.265s; `--re-render` 0.008s, 0 model calls, byte-identical SRT/ASS |
| **D** | Complete French Episode (1379.901s) | **FAIL / BLOCKER in Stage 3b** | 16 fixed chunks, 221 rows, 0 wholly untimed (down from 12); 221/221 context in 1 call (19,359 in / 5,083 out / 2,073 reasoning); 3/6 reference timing support; 1 real `joined_passage_rejection`; **deterministic reproducible `batch[204:216]:ValueError:Invalid comparative review decision`** preserved primary MT and blocked Stage 4 render (confirmed on retry) |
| **E** | Independent Multilingual Control (Mandarin 2-speaker overlap, 10.38s) | **PASS (with honest residual)** | 266.446s wall; 2 speakers separated on 2 ASS lanes, 0 conflicts; Speaker A verbatim CJK match & accurate English; Speaker B kept ASR-artifact nomination unresolved, but English review hallucinated Singapore geography on a misheard foreign toponym (residual) |

## Resource Ownership & Restoration

- Candidate-only CUDA configs (`NEMOVOX_MODEL_RESOURCES`, `NEMOVOX_LIFECYCLE_CONFIG`) used exclusively; owner checkout (`/workspace/nvidia-transcribe`), deployed Studio process, and production configuration untouched.
- Peak GPU VRAM observed: `31,451 MiB` (RTX 5090 32,607 MiB).
- Final GPU state: `3,044 MiB` used / `29,083 MiB` free; both owned systemd user units (`nemovox-asr.service`, `nemovox-text-review.service`) verified `inactive (dead)`; unrelated desktop (`kwin_wayland`) and embedding (`text-embeddings-router`) processes undisturbed.
