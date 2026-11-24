# Clean current-workflow consolidation

## Authority

Operator decision: clean forward development for personal-use Nemovox; see
AGENTS.md. Existing Issue #14 and draft PR #15. No additional PR or issue
required for this consolidation. No merge or deployment authorized here.

The latest successful host qualification tested
604e8dc9c68c13029d3d883a9c2e1a1322b552de, tree
ba9aed1e190a94245918d421d8e6978d56d7a5aa.
Evidence-only head c37e02607e175b23caf43cc820cbf2123942c202.
The uploaded 2,144,130-byte private archive
issue14-audio-source-recovery-20261124T035505Z.tar.gz was independently
SHA256-verified (d55c0f14af02f82e744015e336f6bbdd84ecc510fa09aac3642ca845ce021da1),
22 unique members, 21 matching checksums. The archived source/translation
contracts contain 209 utterances, eight selectively reviewed, two changed
English strings. Host reports 85 focused, 508 core and 508 Studio tests passed.
This is feasibility evidence, not future acceptance authority.

## Design decision

One normal production workflow, not a permanent offline revision framework.
Carry forward capabilities, not every experimental module.

1. Preserve useful ASR completion checks, plausible CTC/overlap evidence,
   speaker-turn handling and honest uncertainty. Consolidate source evidence
   around the current job, not past recording versions.
2. Cheap QC finds material ambiguities. Use light text review for routine
   contextual wording; reserve bounded reasoning and audio-supported
   adjudication for consequential gaps, timing displacement, duplicate
   hypotheses, polarity or unintelligible turns. Add visual evidence only
   where it is likely to resolve a material ambiguity. Audio evidence can be
   independently obtained from the same original-media crop using an admitted
   model; a text-only advisory is not that evidence. No synthetic reviewer
   attestation or episode-specific production repair.
3. Resolve source events before dedicated Riva translation where feasible;
   one canonical source representation. The qualified recovery operation
   demonstrates insertion of audio-attested ASR-absent speech, recovery of
   unowned observations and removal of contradicted output observations.
   Reuse useful logic or simplify it, rather than requiring old recording
   contracts, a previous English track and a manually constructed dossier.
4. Use the already qualified bounded NInfer Messages transport for current
   semantic review and target-language postedit; preserve important meaning
   and polarity. Use sequential managed resources on the single GPU.
5. Render the current canonical track through existing SRT/ASS output.
   Keep Issue #9 renderer-specific optimization separate.

## Consolidation and removal inventory

The exact PR snapshot before this document contained 416 changed files,
including 346 under staging/ (299 evidence paths), 40,094 added staging
lines, 32 source paths and 27 test paths. Most staging additions are not
application functionality.

Central must inspect real imports, entry points, config consumers and tests
before selecting deletion paths. Primary consolidation candidates include
the offline reviewed_revision_runner, reviewed_unit_amendment,
source_fusion_application, source_fusion_downstream, source_unit_lineage,
source_reconciliation, source_event_replay, source_challenger_timing,
source_promotion, evidence_fusion and their tests, plus semantic review queue
and audio recovery adapters where they duplicate the selected canonical
path. This is a candidate inventory, NOT an approved per-file deletion list:
extract needed invariants/functions before deleting their original container.
Remove obsolete CLI commands/aliases, episode validation overlays, experimental
configuration, documentation and fixtures when their current consumers are
gone. Do not retain them solely for prior tests or reproducibility.
Replace current-behavior tests with lean functional regressions where needed;
test count may decrease.

Delete superseded Issue #14 staging evidence/handoffs from the active tree
as part of the consolidation, not by rewriting Git history. Keep only the
controlling current contract and fresh current-version acceptance packet,
or remove the temporary staging directory after acceptance. Do not require
a new archive of old evidence. Host cleanup must use an explicit reviewed
path inventory of disposable generated development outputs and old isolated
worktrees; do not delete original media, installed models, user library
subtitles, user config, active jobs, unrelated workspaces, or uncommitted
operator work. If ownership is uncertain exclude the path. Do not treat
historical outputs as protected acceptance fixtures.

## Clean qualification

No new historical replay or model-wire-only campaign. First consolidate
production code, CLI/Studio wiring and functional regressions centrally.
Then one host qualification from fresh exact-head source and clean output
roots, current config/models, no previous ASR/translation/review checkpoints,
no transplantation of old corrected contracts or episode-specific patches.

Use the same original French episode as a fresh real-media challenge, plus
a small independent different-language/scene control (e.g. Mandarin with
repetition/overlap). Inspect the NEW output for meaningful omissions,
polarity, narrative sequence, false duplicates, important speaker turns,
readability, and material timing defects. Known problem windows can guide
sampling but old word counts, exact old cue text/IDs/hashes, old case
adjudications or 6/7-component outputs are not acceptance targets.
Include genuine repeated speech, silence/music, uncertain speaker and
negation controls, and current CLI/Studio integration, resource restoration,
and useful current-run rerender/resume if retained. Measure current runtime
and actual escalation workload, not an unsupported accuracy guarantee.
One fresh evidence packet tied to current SHA/tree/config/models/media;
no old-run receipt reuse. A lower test count after deliberate removal of
retired behavior is not a failure; meaningful active regressions must pass.
No obligatory human WER/CER/DER target or word-perfect transcript.

This document records the operator-approved development policy and the
next architecture direction, not completed production integration or
deletion authorization for unspecified user files. Central will stage
substantive implementation and exact cleanup inventory before host execution.
