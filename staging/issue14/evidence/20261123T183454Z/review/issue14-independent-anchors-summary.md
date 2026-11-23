# Issue #14 Independent Anchors & Bounded Phase 3b AV Review (`20261123T183454Z`)

## 1. Offline Independent Timing Anchor Replay
- `whisper_large_v3` vs Nemotron CTC cleanly isolated all `4` displaced `W03b` c11 spans (`-5.925s`, `+6.44s`, `+11.43s`, `+10.1s`, MAD `<= 0.34s`) while showing concordant `c10` overlaps within `+0.12s..+0.16s`, leaving `FR_C2_repeat` unmatched, and keeping `no_automatic_source_mutation=True`.
- `whisper_large_v3_turbo` confirmed negative-control divergence: caught `FR_W02a_drift` (`+0.42s` shift) and `FR_W05a_S05`, but hallucinated radio credits over `FR_W04c` fanfare (`2` unmatched).
- Mandarin EP166 negative control emitted `0` candidates (`unit_type=character` skipped cleanly), leaving `41` untimed and `69` unmatched c14 units untouched.

## 2. Direct-Audio Ground Truth Before Vision
- `W02b` (`212.0-236.0s`): direct listening + 100ms RMS confirms `6` clean turns alternating across `2` distinct child voices (`212.05-215.55`, `215.70-218.25`, `218.60-220.05`, `220.20-222.00`, `222.05-231.50`, `231.90-235.85s`), even though both Nemotron (`chunk_002:spk_1`) and pyannote (`SPEAKER_00`) collapsed them to 1 speaker.
- `W04a` (`1201-1256s`): `FR_W04a_early.wav` proves `utt-0187` was pulled `-27.3s` into pure earthquake/rock-collapse SFX; `FR_W04a_mid.wav` confirms real dialogue at `1230.5-1236.9s`; `FR_W04a_late.wav` confirms panting SFX then speech at `1252.0-1255.8s` (`+20s` displacement error in Nemotron).

## 3. Bounded Phase 3b Audiovisual Corroboration (Non-Production)
- **Capability & Isolation**: `nemovox-text-review.service` (`chat`, Qwen3.8-27B NVFP4 `--vision`) passed its 1-image probe (`24` frames counted, HTTP 200) under `ModelLifecycle` (`vram_before=3105 MiB`, `vram_loaded=30675 MiB`, `vram_after=3103 MiB`, `service_state_after=inactive`).
- **Pass 1 -> Pass 2 Findings**:
  - **Corroborated**: `AV_W02b_child_turns` dense 5fps cuts & lip transitions matched child turns 2-3; `AV_S05_pichu_shouts` dense 4fps confirmed animated bird beak movement on the parrot voice (`1370.8-1374.8s`); `AV_ZH_525_handoff` dense 5fps confirmed elder lip change (`523.6-524.4s`) + burned-in subtitle; `AV_W04c_fanfare_trap` confirmed male mouth movement at `1264.1-1266.3s`.
  - **Negative Controls Passed (3/3)**: `AV_W04c_fanfare_trap` (`1261.0-1263.8s`), `AV_S02_H2_boundary` (`359.7-364.4s`), and `AV_ZH_c14_music_trap` (`1260.0-1276.0s`) all returned `false_speech_attribution_detected=false`.
  - **Concrete Failure Modes**: Off-screen speakers (`AV_W02d` opening night ocean, `AV_C2` off-screen female overlap, `AV_S05` narrator/child, `AV_W04c` female question), overlapping speech (`C2`), and animation cut-leads (`W02b` ~`220.9s` cut leading `222.0s` audio boundary by ~1s; `W02b_turn_6` & `W02d_turn_2` visual/audio conflict).
- **Production Integration Decision**: **Do NOT integrate vision into the production ASR/subtitle pipeline.** Vision cannot recover off-screen dialogue, separate overlaps, or replace missing timestamps safely, though it is viable as an offline human/audit diagnostic aid under explicit lifecycle control.
