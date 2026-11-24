# Issue 14 original gate dispositions

The current qualification campaign stopped at the CPU regression gate on candidate `db8faeaa7432e1f53fbbc254b145453b191551dc` (`tree 06cc432bd15fe7a5086a1dd7bab4d412de81960d`) before entering GPU execution.

| Original gate | Disposition | Current evidence |
| --- | --- | --- |
| Temporal drift W01/W02/W03/W04/S04 | Unverified | Not re-run on candidate; blocked at CPU gate before French resume. |
| Duplicate retention W02c/W03a/W03c | Unverified | Not re-run on candidate; blocked at CPU gate before French resume. |
| Coverage W05a | Unverified | Not re-run on candidate; blocked at CPU gate before French resume. |
| Core masking W04b/S03 | Unverified | Not re-run on candidate; blocked at CPU gate before French resume. |
| Clean, repeated and simultaneous controls | Unverified | Focused overlap/displacement CPU tests passed; live French controls unexecuted under CPU gate block. |
| Speaker and turn | Unverified | Blocked at CPU gate before French resume. |
| Source-to-English integrity | Unverified | Structural CPU source-quality/translation/review tests passed; live execution blocked at CPU gate. |
| Quantified before/after | Unverified | Baseline reproduced in prior packet; no full-length candidate output produced. |
| Workflow compatibility | Blocking | `compileall`, imports, diff checks, and 419/420 non-GPU tests passed (including 3 new rechunk recovery tests), but `tests/test_issue14_asr_integrity.py::test_legacy_completed_checkpoint_cannot_hide_length_finish` failed in focused (`279 passed, 1 failed`) and both full non-GPU suites (`419 passed, 1 failed`). French resume and Mandarin Studio control not executed under stop condition. |
