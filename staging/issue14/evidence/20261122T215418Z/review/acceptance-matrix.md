# Corrective A/B Acceptance Matrix (`866aa734`)

| Gate | Status | Notes |
|---|---|---|
| Phase A admission & read-only invariance | **PASS** | `main`=`8f6c03cb`, handoff=`05ac52db`, source=`866aa734`; all inputs & Studio unchanged. |
| Phase B compileall & Studio suites | **PASS** | `compileall=0`; Studio `49 passed`. |
| Phase B focused & full non-GPU pytest | **FAIL** | `test_long_displaced_duplicate_from_adjacent_excerpts_is_not_rendered_twice` failed at line 64 (`displaced_sequence_ctc_unverified` vs `cross_chunk_ctc_disagreement`); all other `273` non-GPU / `31` focused tests passed. |
| Phase C `W04c` & suffix promotion guards | **PASS** | `c14:12` preserved intact; zero unintended suffix promotions (`c12:73`, `c03:299` excluded). |
| Phase C `W05a` & `W04b` boundary repairs | **PASS** | `W05a` recovered `6/6` words (`c14:248..253`); `W04b` selected complete `c14:3..10` cleanly. |
| Phase C `W02c` & `W03a` displaced duplicates | **PARTIAL** | `W02c` removed `34` duplicate words (`10` residual words across inflection/length bounds); `W03a` removed `18` words while keeping interjections & `<8` cross-speaker tokens. |
| Phase C negative controls (`C1`, `C2`, `1327/1329s`, Mandarin/unknown) | **PASS** | Clean speech, genuine repetitions, and interjections `100%` intact. |
| Phase C isolated CTC timing drift (`W01/W04/S04`) | **FAIL (Unchanged Scope)** | `0/2415` quarantined (`63` review-only large gaps). |
