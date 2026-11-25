# Completed Review-Anchor Qualification Evidence (`20261125T214000Z-f45c91d5`)

Sanitized public evidence packet for issue #19 / PR #21 (`issue/19`) qualifying candidate commit `f45c91d5c92297fe8b93d279a27dadd5ba54c391` (tree `a263fdc96fe3a495d88a9f1961a92d88100a0ab6`) against baseline executed source `cabed83dc5685d18acccc354d1519b4e6505cf8e` (tree `99afccd2bc11e1b2036137f1d79aeb94be11535e`), building on prior tested source `e33ff126f93bb61cb000d7e5335d5ed379616ef9` and prior evidence carrier `5f238112209987e6a0e7e6048c40923925522ea4`.

**Result:** `PASS`

No private dialogue text, raw model responses, private source excerpts, or credentials are included in this public packet. Full private per-row deltas, derived contracts, rendered SRT/ASS outputs, raw render CLI logs, and diffs are retained in the host-local private archive:
`review-anchors-f45c91d5.tar.gz` (`1381884` bytes, SHA-256 `6936aa0de1f80e1ade485e80245998195cb1bc168072871b19977180375e2d6f`).
Prior failure evidence archive `review-anchors-e33ff126.tar.gz` (`595243` bytes, SHA-256 `05777ad39f9491f92bda999977a6fcb927477a07c11c1337cd59433b5c38598d`) is preserved untouched.

## Summary

- **Input Equivalence (`e33ff126` == `f45c91d5`)**: Identical `src/` (`058636e6ce4158a4eec69455afb5ad00e25e522b`), `web/` (`1011e5c23e77821de13e8a1043735f792d9d92c7`), `configs/` (`c10b6aaa015ef8f442bff87dd850189631a83aa3`), and all existing tests. Reused prior `e33ff126` qualification receipts (`39` focused, `554` full CPU, `51` Studio, `web/` 14 tests + build from `3e6f8d82`).
- **New Evaluator Qualification**: `compileall` PASS (exit 0), `git diff --check` PASS (exit 0), `tests/test_review_anchor_replay.py` `13 passed` (exit 0).
- **Baseline Replay (`cabed83d`)**: `PASS` (exit 0), `249/249` cache hits, `0` API calls, `0` changed dispositions, exact metric and checkpoint parity.
- **Candidate Replay (`f45c91d5`)**: `PASS` (exit 0), `249/249` cache hits, `0` API calls, `137` accepted review edits total, exact match with prior diagnostic receipts:
  - `bounded`: `changed=16` (`3` text changes restoring joined westward passage + `1` status-only reclassification preserving primary MT under dependent-passage rejection), `unresolved_source_rejections=1`, `dependent_passage_rejections=2`, `comparative_rejections=4`.
  - `mandarin`: `0` changed rows (`changed=1`, `comparative_rejections=1`, unsupported geographic substitution prevented).
  - `french`: `changed=120` (`1` text change recovering sentence-initial preposition capitalization false positive), `unresolved_source_rejections=9`, `rejected_polarity=1`, `comparative_rejections=27`, `dependent_passage_rejections=2`.
- **Offline Public Render & Repeat Rerender (`--re-render`)**: `PASS` (all 3 cases exit 0, zero model calls, translated contracts preserved byte-for-byte):
  - `bounded`: `35` renderable cues (`5` top lane, `0` overflow, `0` unrenderable, `1` conflict); repeat rerender exit 0 with **100% byte-identical** `translated.contract.json`, SRT, and ASS.
  - `mandarin`: `2` renderable cues (`1` top lane, `0` overflow, `0` unrenderable, `0` conflicts); SRT & ASS **100% byte-identical** to baseline.
  - `french`: `295` renderable cues (`48` top lane, `0` overflow, `0` unrenderable, `6` conflicts).
- **Resource State**: CPU-only; `0` GPU inference invocations; GPU memory ~3046 MiB pre/post; `nemovox-studio.service` undisturbed; owner checkout unchanged.
