# Issue 14 completion qualification

**Blocking primary ASR exhaustion after bounded recovery.** Tested application `b3e5a785e849e909b89d45df8aa606f98cde7925`, tree `b50175de4e70d6dd20736443540c6f5fb426b345`.

The first full French CLI attempt exited 1 at chunk 20, inference window 767.665–815.256 s. Both raw generations stopped at 16,384 tokens and had no CTC words. A single same-job resume reused 20 verified complete chunk responses and completed chunk 20 on the same audio. It exited 1 at chunk 28, inference window 1100.141–1154.141 s, after two more raw generations stopped at 16,384 tokens with no CTC words. The run verified 28 of 35 chunks, covering planned cores through 1104.141 s. The ASR guard preserved the four rejected responses and correctly withheld a recording contract and all downstream products. No additional replan or inferred placeholder was used.

The explicit incomplete-workflow stop condition applied. The Mandarin Studio clip was freshly extracted and hashed, but no candidate Studio job was submitted. Live scene completion, joined-passage review, current cue quality and existing-file rerender remain unverified. No output-exhaustion behavior of NInfer was inferred from Nemotron ASR exhaustion.

## Verified structural gates

- Compilation, candidate imports, both Git diff checks, 130 focused tests and 417 full non-GPU tests in each environment passed. Original missing-contract rerender made zero acquisitions. Displaced and short-overlap reproductions passed.
- Archived French edge/interior recovery, Mandarin partial association with the untimed residual, and partial caption status passed against the original verified evidence.
- CPU checks verified normal 8,192-token completion, one 16,384-token retry only on native max_tokens, same evidence/schema/thinking/lease, attempt receipts and 600-second HTTP timeout. Unsupported schema, context exhaustion and admission denial produced no hidden retry, and exhausted partial output was rejected. Joined English passage regression passed.
- Installed NInfer configuration and the latest retained native startup agree on max context 262,144. The tested campaign never started NInfer, so there are no candidate NInfer completion or retry costs.
- The two native French CLI attempts took 280.113 s and 270.669 s. ASR recorded 32 successful HTTP responses including the four model generations that were subsequently rejected. Peak sampled GPU use was 28,237 MiB. Both owned units ended inactive, the original lock was acquirable, and production Studio health and invocation were restored.

The original baseline reproduced 2,415 raw CTC words, 234 utterances, 90 micro-utterances, and 308 ASS events. On exact ASS duration, 63 exceed 18 CPS and 30 exceed 22 CPS. The metadata separately contains 29 high-speed flags. There is no valid same-denominator candidate after comparison. Partial raw source spot checks identify the continuous C1 turn, the repeated C2 exclamation and remaining overlapping W02c hypotheses. These are model observations rather than final subtitles or newly human-verified speech. [Official baseline packet](https://github.com/fvanevski/nemovox/tree/5bf6f37d173cfcc160fdf03b60b051cd0f5e7dea/staging/quality-audit/mcog-s04e01/evidence/20261122T205259Z)

[Original gate matrix](gate-matrix.md) · [Metrics](metrics.json) · [Commands](commands.md) · [Command friction](COMMAND_SHAPE_FRICTION.txt) · [Checksums](SHA256SUMS)
