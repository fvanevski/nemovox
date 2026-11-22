
# Translation Stage Truthfulness & Studio Persistence (`134c4f064b320777640726509ba0d03582f7c395`)

## Six `generate_subtitles` Translation Cases (`tests/test_translation.py`)
All 6 combinations (`openai_chat`, `swiftu` × `ok`, `request_error`, `missing_id`) passed (`6 passed in 0.13s`):

| Backend | Reply Case | `translation` Events | `render` Events | `translated.contract.json` & Output |
|---|---|---|---|---|
| `openai_chat` | `ok` | `running`, `complete` (`detail=""`) | `running`, `complete` | `errors=[]`, `translated_text="Hello."`, SRT written |
| `swiftu` | `ok` | `running`, `complete` (`detail=""`) | `running`, `complete` | `errors=[]`, `translated_text="Hello."`, SRT written |
| `openai_chat` | `request_error` | `running`, `failed` (`completed_with_errors: 1 translation error(s); fallback source text retained`) | `running`, `complete` | `errors` populated, `translated_text="Bonjour."` preserved, `translation_status="fallback_source_after_error"`, SRT written |
| `swiftu` | `request_error` | `running`, `failed` (`completed_with_errors: 1 translation error(s); fallback source text retained`) | `running`, `complete` | `errors` populated, `translated_text="Bonjour."` preserved, `translation_status="fallback_source_after_error"`, SRT written |
| `openai_chat` | `missing_id` | `running`, `failed` (`completed_with_errors: 1 translation error(s); fallback source text retained`) | `running`, `complete` | `errors` populated, `translated_text="Bonjour."` preserved, `translation_status="fallback_missing_id"`, SRT written |
| `swiftu` | `missing_id` | `running`, `failed` (`completed_with_errors: 1 translation error(s); fallback source text retained`) | `running`, `complete` | `errors` populated, `translated_text="Bonjour."` preserved, `translation_status="fallback_missing_id"`, SRT written |

## Studio Failed-Stage Persistence (`tests/test_studio_export_api.py`)
- `test_degraded_translation_event_survives_completed_job` (`PASSED in 0.38s`):
  - `JobRunner._read_stage_events` consumes `NEMOVOX_EVENT` lines (`translation: running -> failed`, `render: running -> complete`).
  - Terminal `GET /api/jobs/{id}` returns `state="complete"` while `stage_history` preserves `[("translation", "failed"), ("render", "complete")]`, `detail="completed_with_errors: 1 translation error(s); fallback source text retained"`, and `finished_at` timestamps without overwriting `translation=failed`.
