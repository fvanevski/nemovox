# Handback: Nemotron Transcription Reliability and Refinement Campaign

**Workspace:** `/workspace/nvidia-transcribe`  
**Starting HEAD:** `889c28189e9418d45cf5de87b979135c5c4dc3c2`  
**Phase A Commit:** `9c7a160` (`Repair eight chunked transcription correctness defects`)  
**Phase B Commit:** `db996420aab2513c324f143586d86b5f86ee0e74` (`Integrate selective Qwen forced-alignment fallback`)  
**Read-Only Reference:** `/workspace/pyannote` (`2b40b1c0c17ce324178af765d8d2c79234870acd`, untouched)  
**Hardware / Server:** NVIDIA GeForce RTX 5090 (32607 MiB), `vLLM` serving `/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4` with verified `quantization=modelopt_mixed`

---

## Executive Decision & Outcome

| Workstream | Disposition | Production Status | Key Quantitative Result |
| :--- | :--- | :--- | :--- |
| **Phase A (`A1`–`A8` Defect Repairs)** | **PASS — Production** | Merged in `src/nvidia_transcribe/` (`9c7a160`) | 8/8 defect classes fixed with unit regressions; fresh episode recovers `221` utterances / `23` speakers (`2064` units) vs prior `213` / `22` (`2016` units) |
| **Phase B (`alignment_fallback = off \| qwen`)** | **PASS — Production Opt-In** | Merged in `qwen_aligner.py`, `_qwen_align_worker.py`, `alignment.py`, `recording.py`, `cli.py` (`db99642`) | Recovers Mandarin OOV `'钾'` (`CER=0.0`, `21/21` `external_qwen_recovered`, `cuda:0`) on `zh_fleurs_01_2006`; rejects live Qwen zero-duration span on `zh_multichunk_synthetic` without clamping (`zero_duration_count=1`) |
| **Phase C (Localized Refinement `C0/C1/C2`)** | **EXPERIMENTAL ONLY** | Isolated in `src/nvidia_transcribe/refinement.py` (`status="proposed_only"`) | **Keep advisory (`status="proposed_only"`):** `C1` suffers `50%` false-correction rate on clean controlled transcripts. Nemotron text-only `C2` failed & stopped; **`/workspace/ninfer/swiftu` `C2a`** achieved **`100%` precision/recall (`0%` FC)** on controlled probes & fixed real homophones (`“跳下垂直近空” -> “桥下垂直净空”`), while independent audio recheck (`C2b`) vetoed text hallucinations (`"chèvres" -> "Cités d'Or"`) |
| **Phase D (Scene-Bounded Diarization)** | **EXPERIMENTAL ADAPTER** | Isolated in `src/nvidia_transcribe/scene_diarization.py` (`authoritative_override_applied=False`) | Scene-fresh `Nemotron-3-Diarization` (`0.07–0.36s` offline / scene, `~428–448 MiB` peak VRAM) matches `3/3` Mandarin multichunk speakers (`consistent_support`) and flags child-voice voice splits (`ambiguous_many_to_one`, 2 advisory flags in `F1_100s`), while confirming persistent music FPs (`O3_music_neg`) — stays supplemental |

---

## Deliverables & Report Index

- Architecture & data-flow boundaries: [architecture.md](file:///workspace/nvidia-transcribe/evaluations/transcription-reliability/architecture.md)
- Eight Phase A correctness repairs & before/after evidence: [correctness-regressions.md](file:///workspace/nvidia-transcribe/evaluations/transcription-reliability/correctness-regressions.md)
- Selective `Qwen3-ForcedAligner-0.6B` fallback evaluation: [alignment-fallback.md](file:///workspace/nvidia-transcribe/evaluations/transcription-reliability/alignment-fallback.md)
- Experimental localized transcript refinement (`C0/C1/C2`): [refinement-evaluation.md](file:///workspace/nvidia-transcribe/evaluations/transcription-reliability/refinement-evaluation.md)
- Experimental scene-bounded `Nemotron-3-Diarization`: [diarization-evaluation.md](file:///workspace/nvidia-transcribe/evaluations/transcription-reliability/diarization-evaluation.md)
- Complete CPU & GPU test verification: [test-results.md](file:///workspace/nvidia-transcribe/evaluations/transcription-reliability/test-results.md)
- Machine-readable campaign metrics: [benchmark.json](file:///workspace/nvidia-transcribe/evaluations/transcription-reliability/benchmark.json)
- Evidence archive manifest: [artifact-manifest.json](file:///workspace/nvidia-transcribe/evaluations/transcription-reliability/artifact-manifest.json)
- Packaged archive: `/workspace/nvidia-transcribe/outputs/handbacks/transcription-reliability-evidence.tar.gz`

---

## Remaining Risks & Open Items

1. **Multilingual lexical accuracy during overlap & proper nouns:** On dense Mandarin overlap (`zh_2spk_overlap`), `C1` audio retranscription improved CER (`0.85 -> 0.6333`) partly by dropping the second speaker stream, and glossary-guided `C1` fixed garbled episode names (`Tincio, Carlos, Diego / Tato -> Esteban, Zia et Tao`) yet introduced hallucinations (`"Enfant du soleil" -> "Enfant du sommeil"`). Any future human-in-the-loop UI should surface `PatchProposal` items (`status="proposed_only"`) for manual review rather than automatic mutation.
2. **Acoustic music false positives in diarization:** Standalone `Nemotron-3-Diarization` still activates on `O3_music_neg` (`627.5–636.5s`), confirming that VAD/music gating would be needed before promoting scene-bounded diarization beyond supplemental evidence.
3. **Dense Mandarin `zh_multichunk_synthetic` remaining untimed char:** `Qwen3-ForcedAligner-0.6B` natively emits `0.08s`-quantized spans and occasionally produces zero-duration spans on rapid CJK speech (`'桥'`); Phase B intentionally preserves `'桥'` as `diarization_bounded_only` (`partial` utterance) rather than fabricating sub-frame boundaries.
