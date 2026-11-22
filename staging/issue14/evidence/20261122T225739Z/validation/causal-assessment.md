# Per-Case Causal Assessment: Baseline `90s` vs Full `pause_aware` (`81704133`)

| Case | Outcome vs Baseline | Causal Layer / Evidence |
|---|---|---|
| **`W01a`** | **PASS** | Verse 1 clean at `4.08–22.29s`; `+55s` phantom name at `82.7s` eliminated; verse-2 singing acoustically unrecovered (`A`). |
| **`W01b` / `W01d`** | **PARTIAL (IMPROVED)** | `-28s` pull of nouns at `49.23s` eliminated (`Les chèvres` now `78.25s`, `Nos héros` `61.79s`); short `7`-tok left-context carryover at `77.64s` stayed below `>=8` cross-speaker threshold (`C`). |
| **`W01c`** | **FAIL (UNCHANGED)** | SFX-masked shout (`~58s`) remains omitted (`A`). |
| **`W02a`** | **PASS** | `-38s` early pull eliminated; timed `239.33–242.43s` (`u025`) with distinct `SPK_05` (surname ASR error without glossary remains `A`). |
| **`W02b` / `W02d`** | **PARTIAL** | Alternating intra-chunk turns still merged (`SPK_04`), but duplicate bleed reduced (`u047` only short `4`-tok cross-boundary echo). |
| **`W02c`** | **PASS** | **All `45` duplicate tokens AND prior `10` residual fragments 100% eliminated** because `c07` `[263.96..307.19]` kept `W02c` inside one core! |
| **`W03a`** | **PASS** | **All `7` residual duplicate fragments + speaker mix resolved** (`SPK_18` priest / `SPK_17` child cleanly separated at `985.8–1007.5s`). |
| **`W03b` / `C2`** | **PASS** | `+14s` forward drift eliminated (`u128` at `1010.75–1014.14s`), `C'est un piège!` separated on `SPK_19`, genuine `C2` repeat verbatim at `1023.46–1026.33s`. |
| **`W03c`** | **PASS** | Duplicate at `1085.5s` eliminated; distinct turns at `1072.26s` & `1085.56s`. |
| **`W04a`** | **PARTIAL (MAIN FIXED)** | Main Q&A (`7+18` words) accurate at `1231.20–1236.94s` (`SPK_24`, `SPK_25`); `c30` left `4`-word `-27s` partial `u145` (`4 < 8` threshold). |
| **`W04b`** | **FAIL (REGRESSED)** | `c31` near `infer_end=1267.70s` interleaved `W04b` (`1253s`) & `W04c` (`1264.5s`) CTC timestamps, shifting `On a réussi,` to `1264.47s` and clipping `Tao! On est sauvés!` across `core_end=1263.70s` (`B/C`). |
| **`W04c`** | **PASS** | `-10s` early split of `Mes` eliminated (`u149` intact at `1264.46–1266.52s`). |
| **`W05a`** | **FAIL (REGRESSED)** | `c34` (`39.78s`) omitted the `6`-word child utterance at `1352.0–1354.0s` in raw ASR (`extended_recovered_units=0` vs prior `6`). |
| **`S01` / `S03` / `S04` / `S05`** | **PARTIAL (IMPROVED)** | `S01` `0.22s` shards eliminated; `S03` accurately transcribed *emmuré*; `S05` recovered previously dropped final clause (`1375.0–1376.9s`, still single `SPK_27`); `S04` clean text (`u154` overlap clamped to `160.0 cps` in renderer `G`). |
| **`C1`** | **PASS** | `29/29` clean words accurate at `687.63–696.68s` (`u089`). |
| **`c28` loop / `utt-0138`** | **FAIL (NEW)** | Forced-cut `c28` (`[1100.14..1154.14]`) hit generation loop (`finish=length`, `16384` tok, `75.2s`), leaving `1` untimed utterance `u138` (`diarization_bounded_only`, `unrenderable_no_usable_timing`). |
