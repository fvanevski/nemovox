# Speaker Reconciliation & Guard Qualification (`81704133`)

## Full-Run Reconciliation (`354` Raw Diar Spans -> `56` Local -> `28` Global, `0` Unknown)
- **Single-Candidate Near-Threshold Guard (`speakers.py:307–318`):**
  - Unit boundary verified: `0.6536` / `0.6637` / `0.6999` blocked (`True`), exact `0.70` / `0.78` allowed (`False`).
  - **Live 23-min full run blocked `6` single-candidate borderline links** cleanly (`0.6571` bidirectional, `0.6639`? wait multi+single: `chunk_002:spk_0`<->`chunk_026:spk_2` `0.6571`, `chunk_012:spk_1`<->`chunk_029:spk_1` `0.6639`, `chunk_020:spk_0`<->`chunk_023:spk_0` `0.6727`, `chunk_031:spk_0`<->`chunk_007:spk_0` `0.6707`) plus `4` multi-candidate near-ties and prevented `1` same-chunk collision (`chunk_011:spk_0`/`chunk_012:spk_1` `0.7277`).
  - Confident positive merges succeeded (`chunk_005:spk_0`<->`chunk_007:spk_3` `0.8757`, `chunk_007:spk_0`<->`chunk_009:spk_2` `0.7819`).
  - Global ID count increased (`23 -> 28`, `27` in utterances) intentionally because `35` smaller chunks yielded `56` local clusters and borderline `0.65–0.67` candidates stayed unmerged rather than falsely collapsing characters. Ground-truth episode speaker count remains unverified.
- **Turn Separation Gains & Remaining Intra-Chunk Limit (`S05`, `W02b`):**
  - `W03b` & `C2` (`1010–1034s`) separated speakers cleanly (`SPK_17`, `SPK_18`, `SPK_19`).
  - `W02b` (`196–236s`) and `S05` (`1354–1377s`): `S05` recovered the previously missing tail dialogue (`u167–u168`), **but Nemotron still emitted `<spk:0>` across alternating turns (`SPK_27`)**, confirming intra-chunk turn collapse still requires a dedicated turn-splitter or diarizer-backed segmentation.
