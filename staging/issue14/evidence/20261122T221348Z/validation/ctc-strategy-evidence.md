# Targeted ASR + CTC Context-Length Evidence (Phase B)

## Admission & Model Fidelity
- **Managed Lease Provenance:** Acquired explicitly Nemovox-owned `nemovox-asr.service` (`NEMOVOX_MANAGED_RESOURCE=1`, `NEMOVOX_RESOURCE_NAME=asr_local`) via `ModelLifecycle.endpoint()` using the exact NVFP4 checkpoint (`quantization=modelopt_mixed`, CTC middleware `model-ctc-bf16.safetensors` verified in journal).
- **VRAM Telemetry:** Baseline `3105 MiB / 29022 MiB free` -> active `26933 MiB / 5194 MiB free` -> post-release `3104 MiB / 29023 MiB free` (`nemovox-asr.service` returned to `inactive`, `startable=true`). Zero model config modifications, zero checkpoint downloads, zero translation/review reruns.
- **Exact Request Parity:** `mode="speaker"`, `source_language="fr"`, `capture_ctc_timestamps=True`, `require_ctc_timestamps=True`, 16 kHz mono WAV, `source_offset_s=<excerpt_start_s>`.

## Comparison: 90s Chunking vs Narrow (`16–24s`) vs Longer (`44–45s`) Windows

| Anchor | 90s+10s Job (`chunk_000..013`) | Narrow Probe (`16–24s`) | Matched Longer Probe (`44–45s`) | Architectural Takeaway |
|---|---|---|---|---|
| `W01a` (`~27–37s`) | Pushed phantom token `+55s` to `82.74s` | `P_W01a` (`24s`): removed `82.74s` push, but mid-sentence cut (`18s`) garbled verse-1 tail and dropped verse-2 singing | `P_W01a_long` (`44s`): restored clean `39`-word verse-1 (`4.08–22.29s`) with no `82.74s` push; verse-2 singing still omitted | Splitting before SFX/silence fixes phantom push; blind mid-sentence cuts corrupt wording. |
| `W01b/d` (`43–63s`, `77–83s`) | Pulled `77.6s` noun to `49.23s` (`-28s`) and pushed `61.8s` clause to `80.3–82.6s` (`+20s`) | `P_W01b` (`24s`) + `P_W01d_late` (`16s`): cleanly separated by silence break into `43.72–63.66s` & `77.64–82.90s` (`<=2s` residual) | Not needed (`W01b/d` resolved at silence boundary) | Separating speech islands across `>8s` pauses stops cross-silence CTC pulls completely; `W01c` SFX shout remains ASR-masked. |
| `W02a` (`240–242.5s`) | `-38s` early pull of partial tokens to `202–203s` | `P_W02a` (`22s`): still pulled `20` tokens `-5.3s` early across `5s` gap onto `234.65–236.50s` (`1.85s` squash), collapsed `4` diar spans to `1` speaker, misheard surname | Not tested in Pass 2 | Counterevidence: even `22s` windows suffer CTC pull across a `5s` internal silence unless split *on pauses*. |
| `W03b` (`1004–1014s`) | `+12..+16s` late drift & merged `<spk:0>` | `P_W03b` (`20s`): fixed `1002–1007.7s` drift, but left `6.1s` gap (`1010.5->1016.7s`), clipped initial clause, and collapsed `4` diar spans to `1` speaker | `P_W03b_long` (`45s`): **zero multi-second drift** across `67` tokens AND restored `3` speakers (`spk_map={0:0, 1:1, 2:2}`) | **Decisive win for pause/turn-aware ~40–45s context**: too-short cuts hurt diarization and alignment; `45s` preserved all turns cleanly. |
| `W04a` (`1231–1237s`) | Bidirectional `-27s` (`1203.25s`) / `+17s` (`1248.5–1250s`) split | `P_W04a` (`22s`): fixed question (`1231.16–1232.84s`), **but completely omitted the `18`-word speaker-2 reply (`1233–1237s`)** | `P_W04a_long` (`45s`): fixed `1201–1204s` grunts, fixed `1231.13–1232.84s` question, **restored all `18` reply words (`1233.05–1236.72s`)**, restored `3` speakers | **Decisive counterevidence against universal `<=22s` chunking**: short crops can drop entire second-speaker turns cleanly captured in `45s`. |
| Controls `C1`, `C2` | Clean (`C1` `29` words, `C2` genuine repeat) | `P_C1` matches within `0.01s` (clip-end at `700.0s`); `P_C2` retains genuine repeat & improves `1030s` wording | Included inside `P_W03b_long` (`1023.42–1026.27s` verbatim) | Both `C1` and `C2` confirm safe re-inference only when boundaries respect speech/pause edges. |

## Recommended Constraints for Central's Next Architecture Step
1. **Do NOT replace `90s` with fixed `15–22s` cuts:** Fixed narrow windows sever clauses (`P_W01a`, `P_W03b`), drop multi-speaker context (`P_W02a`, `P_W03b`), and can lose whole replies (`P_W04a`).
2. **Pause-bounded ~30–45s windows or subinterval VAD splitting across long silent/SFX gaps (`>=3–5s`)** address both root failure modes (`W01`, `W03b`, `W04a`) without sacrificing speaker context.
3. Even shorter windows cannot invent acoustic signal under loud SFX (`W01c`) or supply domain proper nouns without explicit glossary/biasing support (`W02a` surname).
