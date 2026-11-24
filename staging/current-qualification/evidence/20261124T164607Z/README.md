# Issue 14 archived-observation and bounded qualification

Application commit `31f7c4848f3740c4df2bd5619f90455ed64da946`, tree `ee31d0f8d2d760984859699fbd86b014ceeb2b75`. All tests and live runs used these application bytes. The evidence publication only adds review artifacts.

## CPU and archived selector

- **PASS.** Compilation and candidate imports succeeded. The focused suite passed 127 tests. Both full non-GPU suites passed 414 tests. The original missing-input rerender and displaced/overlap reproductions passed, with zero managed ASR acquisitions on the missing-input path.
- **PASS, French crop edge.** The exact archived production replay recovered `src-f0dd3eadf2e8cc2aafb64cbc` at original 792.311–794.851 s. Interior `challenge-0-1:0` supplied approximate timing. The separately extracted boundary observation contributed corroboration only. A different archived French target correctly rejected a conflicting edge sequence.
- **PASS, Mandarin partial association.** The exact replay recovered two approximate spans at 238.820–247.260 s and 247.840–255.640 s using `challenge-0-1:8` through `:17`. Residual `utt-0012` stayed in sequence with no timestamps. Other proposals remained ineligible for no bracketed observations, no interior observation, or insufficient support and an existing competing row. The metrics list the exact recovered identities and remaining predicates.
- All 176 members of the prior archive and its outer SHA-256 were verified before replay. Archived evidence was used only by the CPU diagnostic.

## Bounded assessment

- **Fresh French occurrence recovery improved.** The 690–900 s CLI scene recovered a timed turn at 792.618–795.157 s through two Nemotron crops, then passed it to normal scene adjudication and Riva. Initial scene review received five preceding and seven following turns. Expanded review received 11 and 14 with no budget omissions. It switched from a tentative alternative about going too close to retaining a warning interpreted as going out for air. Current alternatives still disagree about the action. The independent challenger was selective and did not retest the already-represented occurrence. Exact wording and speaker identity remained unverified.
- **Full-episode quality gate failed in comparative English review.** In the fresh French scene, the source-only request correctly linked an object split across two target rows. The comparator received both complete passages but rejected the correct proposal, treating one proposal row in isolation and claiming the primary continuation was coherent. The subtitle retained a materially wrong patient and an orphaned object. The first reasoned translation improved here, but the final English did not. Neither the full French episode nor the separately gated Studio control ran.


- **Fresh Mandarin timing remained unresolved.** A different ASR segmentation left one substantive turn untimed and placed another long turn in a 0.137-second interval. Both Whisper crops provided ten eligible observations across the sequence. The selector rejected nonunique segment-to-row overlap. This different integration failure does not invalidate the archived partial-association pass.
- **Caption status passed on a real archived positive case.** Applying the earlier complete caption response now preserves the event bounds and marks the source as partial. On the fresh Studio case, NInfer read the caption in four in-event images. The fused request reached its 4,096-token output limit while repeating context IDs. The application retained source and surfaced the incomplete assessment. Fresh partial application remains unverified.


- **Wire and positive controls passed.** Verified 16 reasoned source-first requests with the primary English excluded and 16 paired comparisons with every target row. The French control repaired the idiom and the Isabella pronoun, and preserved the restrictive meaning. Mandarin review restored past-time chronology and rhetorical impossibility. Some awkward wording and source noise remain.
- **Execution and lifecycle passed after bounded retry.** Both first admission failures are retained. Fresh Studio and same-job French resume then completed. All six exports stayed within the actual clip durations. Existing-file offline rerender reproduced both outputs with no new inference. Native measurements include 58 Messages requests totaling 462.0 s on the wire across all attempts and 31,516 MiB peak sampled GPU use. Both owned units relinquished the original lease, and production Studio health was verified.

[Metrics](metrics.json) · [Commands](commands.md) · [Command friction](COMMAND_SHAPE_FRICTION.txt) · [Checksums](SHA256SUMS)
