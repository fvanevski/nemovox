# Speaker Merge Guard & S05 Qualification (`4c6c7ac5`)

## Single-Candidate Low-Margin Guard (`src/nvidia_transcribe/speakers.py:307–311, 556–569`)
- **Historical problem pairs (`5bf6f37d173cfcc160fdf03b60b051cd0f5e7dea:staging/quality-audit/mcog-s04e01/evidence/20261122T205259Z/`):** `chunk_001:spk_0` <-> `chunk_011:spk_0` (`cosine=0.6637`) and `chunk_002:spk_2` <-> `chunk_014:spk_3` (`cosine=0.6536`) previously merged unconditionally when `len(nbrs) == 1`.
- **Observed guard behavior at `threshold=0.65`, `ambiguity_margin=0.05`:**
  - `0.6536` -> **BLOCKED (`True`)** (`single_candidate_near_threshold`)
  - `0.6637` -> **BLOCKED (`True`)** (`single_candidate_near_threshold`)
  - `0.8754` (`chunk_002:spk_1`<->`chunk_003:spk_1`) & `0.8897` (`chunk_000:spk_1`<->`chunk_014:spk_2`) -> **ALLOWED (`False`)** (positive controls preserved!)
  - **`0.70` Exact Boundary Defect (`FAIL`):** `_near_threshold_single_embedding_candidate(0.70, 0.65)` returns **`True`**, causing `tests/test_issue14_speaker_review_guards.py:45` to fail, because IEEE-754 float `0.65 + 0.05 == 0.7000000000000001 > 0.70` (`0.70 - 0.65 == 0.04999999999999993 < 0.05`). Round/tolerance-aware comparison or exact boundary semantics needed.

## Archived `S05` / `W02b` / `W03b` Speaker Context (From Read-Only Archive)
- Full-episode TitaNet + ASR rerun was stopped by Phase A CPU failure. From `d50a81f6f74b70fef52bbd7b9f39b470019a4e8e:staging/issue14/evidence/20261122T221348Z/`, archived `S05` (`chunk_015`) raw diarization separated `[1354.26–1370.44] spk=0` from `[1371.53–1374.36] spk=1` even though sparse `<spk:0>` output collapsed both and omitted `>1370.63s` tail words.
