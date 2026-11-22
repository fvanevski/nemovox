# Phase A Verification: ASR Completion-Integrity Gate (`4c32acf6`)

## Source Isolation & Executor Binding
- **Isolated detached worktree:** `<ISOLATED_WORKTREE>` at `2d6d0624e4a4a640ee82603f5f346bfcb378918f` (`diff vs 4c32acf6 -- src tests web` empty).
- **Verified import origins:** `<ISOLATED_WORKTREE>/src/nvidia_transcribe/{asr_integrity,recording}.py`.
- **Web diff vs `81704133`:** Empty (`web/` unmodified; reusing `14` Vitest + `tsc && vite build` PASS from `81704133`).

## Deterministic Suites
| Suite | Result | Exit | Skips / Warnings |
|---|---|---|---|
| `compileall -q src tests scripts` | **PASS** | `0` | `0` skips / `0` warnings |
| Focused (`test_issue14_asr_integrity.py` + pause/speaker/timing/chunking/utterances/recording/review) | **PASS (`64 passed`)** | `0` | `0` skipped; `1` `anyio` `PytestRemovedIn10Warning` |
| Full non-GPU (`pytest -q -m 'not gpu'`) | **PASS (`294 passed`)** | `0` | `0` skipped; `1` `anyio` `PytestRemovedIn10Warning` |
| Studio (`.venv-studio` `test_studio_{exports,export_api,api}.py`) | **PASS (`50 passed`)** | `0` | `0` skipped / `0` warnings |

## Fail-Closed Checkpoint & Retry Invariant Verification (`PASS`)
- **`tests/test_issue14_asr_integrity.py`:** All `5` tests passed (`length` with valid CTC rejected as `incomplete`, short repeated `Non! Non! Non!` preserved as `complete`, `>=2048`-char pathological loop under `stop` rejected, legacy checkpoint replay re-audited and rejected cleanly, failed retry preserves `rejected/<sha>.raw.json`).
- **Two-Chunk Synthetic End-to-End Check (`PASS`):**
  1. `chunk_000` returned `stop` (`completed`); `chunk_001` returned `finish=length` (`raw=21000+` loop chars).
  2. `process_media` raised `IncompleteASRGeneration`, wrote `asr_run_status="incomplete_generation"`, retained `chunks/rejected/chunk_001_<sha16>.raw.json`, and did **not** emit `recording.contract.json`.
  3. On retry, `process_media` **resumed `chunk_000` (`resumed_chunks=1`)**, re-ran `chunk_001` cleanly without trusting rejected raw (`calls=['chunk_000.wav', 'chunk_001.wav', 'chunk_001.wav']`), published `asr_run_status="completed"`, and retained the rejected raw file.
