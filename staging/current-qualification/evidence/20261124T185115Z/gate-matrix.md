# Issue 14 original gate dispositions

Campaign stopped at the CPU regression gate on candidate `df82b900ce1ec859530d1e9858ab7a5f4ee74213` (`tree a75fb56c483a1bea0431f472d182698d3449b826`) before live French continuation. Prior live `eb5bc5243677ede15e7b04c836cfabf84d260955` evidence remains in `20261124T183928Z`.

| Original gate | Disposition | Current evidence |
| --- | --- | --- |
| Temporal drift W01/W02/W03/W04/S04 | Residual limitation (prior `eb5bc52`) | Not re-run on `df82b90` due to CPU gate block; prior `eb5bc52` canonical source preserved unchanged. |
| Duplicate retention W02c/W03a/W03c | Residual limitation (prior `eb5bc52`) | Not re-run on `df82b90` due to CPU gate block. |
| Coverage W05a | Pass (prior `eb5bc52`) | Full French ASR and canonical source preserved unchanged from `eb5bc52`. |
| Core masking W04b/S03 | Residual limitation (prior `eb5bc52`) | Not re-run on `df82b90` due to CPU gate block. |
| Clean, repeated and simultaneous controls | Pass (prior `eb5bc52`) | Prior Mandarin Studio control (`00360c7e88554e57a7aa171254c9cf11`) and `eb5bc52` French canonical controls preserved unchanged. |
| Speaker and turn | Residual limitation (prior `eb5bc52`) | Not re-run on `df82b90` due to CPU gate block. |
| Source-to-English integrity | Blocking | New compact-review and deferred-rendering CPU tests passed (`test_translation_review.py`), but `test_programmatic_postedit_rejects_legacy_primary_translator` failed in `subtitle_pipeline.py:159` (`FileNotFoundError` before `ValueError`). Live French continuation not started. |
| Quantified before/after | Unverified (final French SRT/ASS) | French SRT/ASS not rendered due to CPU gate block on `df82b90`. |
| Workflow compatibility | Blocking | `427/429` non-GPU tests passed in both `.venv` and `.venv-studio`, but 2 existing config-order `ValueError` tests (`test_programmatic_postedit_rejects_legacy_primary_translator` and `test_programmatic_riva_condensation_requires_explicit_text_resource`) failed due to `rc_input.is_file()` raising `FileNotFoundError` before config validation in `subtitle_pipeline.py:159`. |
