# Issue #5 Final Exact-Head Qualification Findings

- **Tested SHA:** `d23293ae87e27c00ee798d73d762efb13fd4bb3e` (clean isolated worktree, zero local modifications)
- **Blocking findings:** None.
- **Remediation checks verified:**
  - Qwen parser default (`--qwen-device` omitted -> `None`, explicit `--qwen-device cpu` -> `'cpu'`)
  - Skipped-condensation credential guard (`--translator none --condense-high-cps` does not require unset condensation `api_key_env`)
  - Riva condensation incompatibility guards (`generate_subtitles` programmatic guard & `condense_high_cps_cues_with_text_model` guard)
  - Empty higher-precedence env override fail-closed (`TEXT_BASE_URL=""` / `TEXT_MODEL=" "` do not silently fall back to `SWIFTU_*`)
  - Legacy SwiftU options/env/aliases & single-clip ASR role default (`http://127.0.0.1:8190/v1`) + legacy explicit override (`http://127.0.0.1:8000/v1`)
  - Resource-sensitive cache keys (refinement, translation, condensation, ASR `_config_signature`)
- **Documentation audit (`README.md`, `AGENTS.md`, `web/README.md`, `docs/gui/ARCHITECTURE.md`, `docs/model-resources/CONFIGURATION.md`):**
  - No blocking inaccuracies against CLI `--help`, `model_resources.py`, `studio_api.py`, `web/src/api.ts`, or `web/src/App.tsx`.
  - Nonblocking note: `CONFIGURATION.md` line 181 describes the UI dropdown options (`openai_chat`, `riva`, `swiftu`), while Pydantic `Options` also includes `'none'` for `transcription` tasks (rejected with 422 on `translation`/`subtitles`).
- **Live host endpoints:** `127.0.0.1:8190` and `127.0.0.1:8001` remain unavailable (`Connection refused`); `TEXT_LIVE_SMOKE=UNVERIFIED`, `ASR_LIVE_SMOKE=UNVERIFIED`, `GPU_SERVICE_MUTATION=no`.
