# Issue #14 qualification handback

**HOLD — CPU gate failed.** The required CPU suites and web build passed. Two additional deterministic checks reproduced a consequential source chronology decision and an offline rerender violation. The handoff stops qualification at this point.

| Gate | Result | Evidence |
| --- | --- | --- |
| CPU | FAIL | All required suites passed. Source selection and rerender fail the invariant checks in [reproduce.py](reproduce.py). |
| CLI | FAIL | The default and alternative translator settings passed. A rerender with original media and a missing translated contract reached managed ASR acquisition. Fresh original-media generation is BLOCKED. |
| Studio | BLOCKED | The CPU suites, frontend build, and synthetic subprocess-boundary checks passed. The new real-media Studio job is blocked by the CPU gate. |
| Source quality | FAIL | A later corroborated turn crossed an intervening speaker turn and was recorded as already represented. See below. |
| Translation and review | BLOCKED | Synthetic pipeline coverage passed for canonical source reaching the translator and bounded Messages review configuration. Live Riva input and English review remain unverified. |
| Renderer | FAIL | SRT and ASS were regenerated from a current synthetic translated contract with model tripwires. The missing-contract rerender violates the inference-free requirement. Real cue quality remains blocked. |
| Runtime and resource lifecycle | BLOCKED | ASR and text ownership markers matched and both units were inactive at both observations. Live start, release, CTC, quantization, VRAM peak, request counts and timing remain unverified. |

## Source selection reproduction

The synthetic recording has SPK_A saying “On y va !” at 10.00–12.00 s and SPK_B saying “Attends !” at 11.30–11.60 s. Observation A:0 places “On y va !” at 11.80–13.80 s from the 7–18 s crop. Observation B:0 places it at 11.90–13.90 s from the 5–16 s crop. The later hypothesis starts after the entire intervening turn and differs from the earlier source start by 1.80 s. The two new hypotheses pass the candidate's two-context corroboration gate at its default 0.85 s tolerance.

The selector returns only the two original rows and records A:0 / B:0 as `already_represented`. In [source_quality.py](../../../../src/nvidia_transcribe/source_quality.py#L282), the exact-text shortcut uses interval overlap greater than 0.08 s without requiring correspondence in start or end time. The competing hypothesis therefore contributes no unresolved decision to the report. Retaining the original times may be correct. The available evidence permits a displaced timing hypothesis or another occurrence, and the ambiguity needs to stay visible. These are synthetic ASR observations with fictitious identifiers, so this establishes the decision logic without asserting an auditory finding.

## Offline rerender reproduction

Invoke `generate-subtitles <existing synthetic media> --output-dir <empty root> --source-language fr --re-render`, intercepting managed ASR acquisition. The CLI returns 1 from the tripwire after attempting acquisition once. The same flag with a current translated contract renders SRT and ASS with no acquisitions, and its contract bytes remain identical.

The shortcut in [subtitle_pipeline.py](../../../../src/nvidia_transcribe/subtitle_pipeline.py#L195) requires both the rerender flag and an existing translated contract. When the file is absent, execution enters the normal model workflow. The previous main revision has the same branch condition. The offline contract requires an early missing-contract error before stage acquisition.

## Identity and disposition

- Tested and final application source commit `6ce71c5ef14aa7960e926887f3e588d5689edb0b`, tree `3aa7d84d53f015e3d8bde3f7e0d73d0c836a5114`.
- The branch and open draft PR #15 were authenticated at that head before the run. This carrier adds evidence files only. [Draft PR](https://github.com/fvanevski/nemovox/pull/15)
- Candidate merge base `8f6c03cbc70e84bf1970ba364207a02c9fe22b44`. The full active delta is available from Git. The 42 changed paths are listed in [candidate-files.txt](candidate-files.txt). [Exact source comparison](https://github.com/fvanevski/nemovox/compare/8f6c03cbc70e84bf1970ba364207a02c9fe22b44...6ce71c5ef14aa7960e926887f3e588d5689edb0b)
- The candidate was clean in a detached worktree. The operator's main checkout and checked-out issue branch were preserved. No host rollback action was needed. Live French and control media selection and processing were gated before any new media was opened.
- [Verification commands and observed exits](commands.md) · [Runtime and config fingerprints](runtime.json) · [Command shape friction](COMMAND_SHAPE_FRICTION.txt) · [Manifest](manifest.json)
