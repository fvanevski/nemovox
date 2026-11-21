# Cross-Chunk Speaker Reconciliation & Sortformer Comparison

## 1. Upstream Sortformer Arrival-Order Speaker Cache vs Stateless vLLM Chunking

NVIDIA's underlying Nemotron 3 / Sortformer streaming diarizer uses an **Arrival-Order Speaker Cache** across internal acoustic windows in a single continuous pass. Inspection of our archived single-pass 23-minute French response (`outputs/responses/longform/fr_full_episode.raw.json`) confirms this split behavior:

- **Acoustic Sortformer `diarization` stayed coherent:** `229` continuous segments over `7` speaker IDs (`'0'`–`'6'`).
- **Autoregressive LLM `<spk:N>` tags collapsed over long context:** `1992` words exhibited **`481` word-level speaker switches**, switching speakers inside individual phrases.
- **vLLM architectural boundary:** `vLLM`'s OpenAI-compatible serving endpoint handles each chunk HTTP request independently and does **not** expose persistent Sortformer speaker-cache state across separate chunk requests. Consequently, even though Sortformer runs cleanly inside each `90s + 10s` request (yielding `8`–`49` diarization segments per chunk in `fr_full_episode_90s`), chunk-local speaker IDs (`spk_0`, `spk_1`, ...) must be reconciled externally across chunks.

Also notable: comparing `chunks/chunk_002.raw.json` (`180`–`270s`, which covers `03:16–03:55`) showed Sortformer detected `5` diarization speaker IDs (`'0'`–`'4'`), while the LLM text only emitted `3` speaker tags (`'0'`, `'1'`, `'2'`), folding Esteban and Tao into a single local speaker. Retaining raw Sortformer diarization intervals alongside reconciled LLM utterances ensures downstream tools can audit this discrepancy without re-running inference.

## 2. Two-Stage Conservative Reconciliation Design (`src/nvidia_transcribe/speakers.py`)

Each local speaker cluster is first normalized as `chunk_XXX:spk_Y` (with unattributed tags explicitly kept as `chunk_XXX:unknown` and **never** collapsed into `SPK_00`). Simultaneous co-occurrence between any two local speakers within the same chunk builds a strict **`cannot_link` constraint** (5 concurrent overlap pairs recorded in `fr_full_episode_90s`, up to `8.51s` overlap in chunk `3`).

Reconciliation runs in two ordered stages with union-find merging blocked by `cannot_link`:

1. **Stage 1 — Exact Adjacent Boundary Matching (`temporal_boundary_match`):**
   - Inspects the `[boundary - context, boundary + context]` overlap window between chunk `k` and chunk `k+1`.
   - Computes combined score from Sortformer diarization IoU and boundary token text/time overlap, applying Hungarian/greedy bipartite assignment (`score >= 0.35`, with close runner-ups logged in `ambiguous_matches` rather than auto-merged).
2. **Stage 2 — CPU Speaker Embeddings (`TitaNetSpeakerEmbedder`, optional):**
   - Uses NeMo's built-in `EncDecSpeakerLabelModel` (`titanet_small`, `192`-dim L2-normalized embeddings) executed **strictly on CPU** (`0 MiB` GPU VRAM), averaging up to 3 high-confidence single-speaker speech crops (`>= 1.2s`, non-overlapping) per local cluster.
   - Links non-adjacent ABA reappearances across chunks (`cosine >= 0.65`) only if neither merged group violates `cannot_link` or merges already distinct speakers seen in the same chunk.

## 3. Empirical Comparison: `temporal_only` vs `temporal + TitaNet` (Full 23-Min French Episode)

Source data: [`benchmark.json`](file:///workspace/nvidia-transcribe/evaluations/chunked-core/benchmark.json) (`french_full_episode` vs `french_full_episode_temporal_only`).

| Mode | Raw Local Speakers | Global Speakers | Unresolved (Single-Chunk Provisional) | Boundary Temporal Links | TitaNet Embedding Links | Ambiguous Matches | Collisions Prevented (`cannot_link`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `speaker_embeddings="none"` (`temporal_only`) | 44 | 28 | 18 | 16 (`weighted`) | 0 | 2 | 0 |
| `speaker_embeddings="titanet"` (`threshold=0.65`) | **44** | **22** | **14** | **16** | **6** | **2** | **3** |

Key observations from `fr_full_episode_90s/recording.contract.json`:
- **Cross-chunk ABA recovery:** TitaNet re-linked `6` non-adjacent clusters across the episode (e.g., `chunk_000:spk_1` ↔ `chunk_014:spk_2` into `SPK_01`, `chunk_002:spk_0` ↔ `chunk_013:spk_2`, `chunk_000:spk_3` ↔ `chunk_011:spk_0`, `0.6637`–`0.6958` cosine similarity).
- **Safety under animated voice acting:** `cannot_link` explicitly prevented `3` high-similarity false merges (e.g., `chunk_002:spk_1` vs `chunk_003:spk_3` at `cos=0.8754`, `chunk_002:spk_1` vs `chunk_009:spk_2` at `cos=0.7058`, and `chunk_006:spk_1` vs `chunk_009:spk_2` at `cos=0.6794`) that would have conflated dialogue partners.
- **Explicit uncertainty (`14` provisional clusters):** Brief interjections and 1-word artifact speakers stay labeled `"provisional_single_chunk"` in `speakers[*].identity_status` and listed in `speaker_reconciliation.unresolved_speakers` instead of being forced into false merges.
