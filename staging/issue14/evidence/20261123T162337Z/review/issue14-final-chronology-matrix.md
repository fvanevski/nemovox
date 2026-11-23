# Issue #14 / PR #15 Final Chronology Acceptance Matrix

- **Admitted PR Head:** `f3af69feb470bc6ce2dd1ffd5eff5c8802cd91c6`
- **Central Source SHA:** `8c0244bc22176b8e5f1ce2d2e6e3480505d9ce4e` (`tree=cc3cb0c1721fcc416d308d8392b789cdc7b63801`, `SOURCE_DIFF=none`)
- **Base Main SHA:** `8f6c03cbc70e84bf1970ba364207a02c9fe22b44`

| Gate / Scope | Status | Summary |
|---|---|---|
| CPU & Static Gates | **PASS** | `compileall=0`, focused `112 passed`, full non-GPU `400 passed`, Studio CPU `50 passed`, `git diff --check=0` |
| W04c & H2 Re-Anchored Audio Review | **PASS** | New bounded WAVs listened & SHA-bound (`W04c [1264.3,1266.3]`, `H2 [364.7,366.2]`); H2 linked to `utt-0074` (`4:24`); old crops & stale revisions rejected |
| FR & ZH Immutable Lineage + Promotion | **PASS** | FR `2051/2051` matched, `H1` echo & `W04c` partial held; ZH `2832/2901` matched, `turn1` held on partial `utt-0062`, `41` untimed kept |
| 22 Indexed FR Cases | **PARTIAL (expected residual ASR/CTC scope)** | `W05a` 6/6 & `W04b` & `W04c` & `C1/C2` pass; `W01a/b`, `W03b`, `S04` CTC drift and `W02b/d`, `S05` diarization collapse remain open upstream |
| Fresh 23-Min French Default ASR Run | **PARTIAL (failed closed on attempt 1, 16/16 on resume)** | Attempt 1 hit `generation_limit_exhausted` at `chunk 12` (fail-closed, rejected raw archived); resume completed `16/16` (`2450` raw units, `217` utterances, `W05a` 6/6 auto-recovered, `S02` lost across `c03/c04` boundary) |
| Riva `fr->en` Subtitles & Rerender | **PASS** | Default `217/290`, Central vetted `234/309`, Agent vetted `236/311`; resume & `--re-render` exact SHA match; Studio source-bound smoke `202 queued` |
| Host GPU & Service Non-Mutation | **PASS** | `3105 -> 3105 MiB` GPU baseline restored; all host services/jobs/models untouched |
