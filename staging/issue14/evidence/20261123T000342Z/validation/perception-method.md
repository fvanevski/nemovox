# Acoustic Perception Method and Held-Out Sampling Protocol

- **Run ID:** `20261123T000342Z`
- **Perception Status:** `ACTUAL_INDEPENDENT_LISTENING_VERIFIED`
- **Method:** Direct multimodal audio audition via native `audio/wav` file inspection on bounded `16 kHz` mono PCM WAV excerpts (`18` French `mcog_s04e01` excerpts totaling ~327 s + `2` Mandarin `ep166_zh` excerpts totaling 36 s), strictly separated from model outputs prior to held-out sampling. Model CTC timestamps, Nemotron diarization spans, and ASR transcripts were NEVER treated as listening truth.
- **Supplemental Objective Timing Calibration:** Deterministic `-32 dB` RMS 0.1 s acoustic energy envelopes were computed on extracted WAVs strictly as objective onset/offset timing anchors for events confirmed by ear (energy thresholds alone are never classified as speech vs SFX).
- **Private Cue Reference Digest:** `private_acoustic_reference.json` SHA-256 `30292212c209241fca4914fe2d41410c844ab7db5a680d912ba6aea7e1460545` (held locally in `chmod 700` storage; excluded from Git to comply with copyright and path-privacy rules).

## Deterministic Held-Out Sampling

- **Seed formula:** `SHA-256(fr_video_sha256 + "issue14-heldout-audio-v1")`
- **Video SHA-256:** `0e77ce3c809e498b97ffe48b53575c29e10d8b08a039b78e2150cd0c845c81b0`
- **Derived seed prefix:** `cd7b6a9764ccc95f`
- **Candidate grid:** `20.0 s` non-overlapping grid slots across `[100.0, 1280.0]`, excluding all named priority (`A`, `B`, `C`, `D`) and control (`C1`, `C2`, `song`, `sfx`) bands (`33` eligible slots total).
- **Deterministically selected 4 intervals (locked before reading any held-out ASR observation):**
  1. `H1_120_140`: `[120.0, 140.0]`
  2. `H2_360_380`: `[360.0, 380.0]`
  3. `H3_900_920`: `[900.0, 920.0]`
  4. `H4_920_940`: `[920.0, 940.0]`

## Per-Subregion Summary of Priority A (`c28` 1099-1156s)

| Subregion | Acoustic Truth | Speech Events | Distinct Voices | Expected Words (approx) | Model Coverage Summary |
|---|---|---:|---:|---:|---|
| `1099-1110` | Impact SFX (1099-1105.5s), urgent high-pitched shout (1105.5-1110s) | 1 | 1 | ~4 | ORIG16 c12: 2w; FULL35 c27/c28, P1, F1: 0w |
| `1110-1123` | Continuation shout/scream (1110-1112s), crash/groans + villain cackle (1112-1123s); NO omitted long speech | 1 | 2 (1 vocalizer + 1 laugh) | ~3 | FULL35 c28: 4 unpop diar spans (`length`); P1: 2 single-word exclamations (`stop`); F1: 0w |
| `1123-1144` | Horse neigh/laugh/music + CRISP 9w adult male urgent speech (`1131.5-1136s`) + child shouts (`1141.5-1144s`) | 3 | 3 | ~14 | ORIG16 c12: 10w; P2: 20w (includes 4 hallucinated interleaved `spk_1` words on `1131.5-1136s`); FULL35 c28: 0w (`length`); F1: 1w (`stop`) |
| `1144-1156` | Urgent repeated male call (`1144.5-1147s`), brief child cry (`1147-1150.5s`), crisp 4w male instruction (`1151-1154s`) | 3 | 2 | ~8 | ORIG16 c12 & P2: 8w; FULL35 c29 (`infer_start=1146.14s`): missed 1144.5-1146s, distorted left context, captured final 4w (`1152.3-1154.5s`) |
