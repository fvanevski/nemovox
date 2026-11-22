# PR #8 / Issue #7 Verification Findings

## Summary
- **Result:** `PASS` (CPU/frontend deterministic qualification and behavioral invariant audit on exact `tested_source_sha=e824321d5680e61044cdbccc9564ca25d2b2f9e9`).
- **Source changes:** `no` (zero edits to production source, tests, configs, or docs).
- **Live GPU/model inference:** `UNVERIFIED` (not exercised per contract; existing GPU services were not altered).

## Verification Coverage
1. **Python compile & CPU tests:**
   - `python -m compileall -q src tests experiments` -> exit 0.
   - Focused suite (`test_translation_review.py`, `test_studio_api.py`, `test_model_resources.py`, `test_resource_integration.py`, `test_translation.py`) -> **76 passed** (exit 0).
   - Full CPU suite (`pytest -q -m "not gpu"`) -> **177 passed** (exit 0).
2. **Studio frontend (`web/`):**
   - `npm test` (`vitest run`) -> **4 test files, 10 passed** (exit 0).
   - `npm run build` (`tsc --noEmit && vite build`) -> exit 0.
3. **Behavioral invariants & six GUI/pipeline corrections:**
   - `studio_api.Options` retains persisted-job compatibility (`refinement="off"`, `translator="openai_chat"`, `translation_review="off"`, `condense_high_cps=True`), while frontend `INITIAL_OPTIONS` uses `refinement="off"`, `translator="riva"`, `translation_review="balanced"`, `condense_high_cps=false`. Studio API rejects non-off `refinement` on `/api/jobs` (HTTP 422) and CLI retains `--refinement` as an explicit legacy opt-in.
   - Audio track selection uses FFprobe audio-only ordinal indices (`media.inspect_audio_streams`); invalid indices, unauthorized paths, and pre-download URL/contract inspects are rejected server-side; frontend resets `audio_stream` on source switch and scopes React Query cache by `(kind, value)`.
   - Bounded read-only FFprobe on two approved single-track `.webm` media files returned accurate single-stream metadata (`index: 0`, `container_index: 1`, `codec: opus`, `language: eng`, `channels: 2`, `sample_rate_hz: 48000`). No approved multi-track local file was present in workspace roots (multi-track parsing remains verified via synthetic FFprobe unit test `test_audio_stream_parser_uses_audio_only_indices`).
   - Translation pipeline preserves raw `source_text` unchanged (`refinement=off` makes zero text-model calls and overwrites any stale `refined_text` with `source_text`), invokes Riva primary MT first, then runs a single English post-edit pass receiving untouched `source_text` and original `machine_translated_text` (never `refined_text` or previously edited English across batches), preserves `machine_translated_text`, fails closed on malformed/incomplete review or Riva worker results, invalidates review cache on model/style/glossary/source/MT changes, keeps condensation off by default, and re-renders offline without model calls.
   - `ExplainedSelect` exposes all four ASR modes (`speaker`, `multispeaker_sot`, `verbatim`, `rich`), English review profiles, and localization styles with hover, keyboard (`ArrowUp`/`ArrowDown`/`Home`/`End`/`Enter`/`Space`/`Escape`), and touch/click descriptions; optional `mcog_fr.yaml` and `a_will_eternal_zh.yaml` glossaries are off by default and preview titles, characters, terms, and notes.

## Non-Blocking Observations (No `BLOCKING` or `IMPORTANT` Regressions)
- `web/src/App.tsx:411`: Glossary field hint still reads `"Optional series-specific names and terms for source correction and English post-editing..."`, although Studio disables source correction (`refinement != "off"` returns HTTP 422).
- `src/nvidia_transcribe/studio_api.py:492`: Subprocess stage progress label reads `"Refining, translating and rendering"` even when `--refinement off` is passed.
- `pyproject.toml:10-20`: `pyyaml` is imported by `studio_api.py:597` and `refinement.py:334` for `.yaml` glossaries without an explicit declaration in `pyproject.toml` dependencies/`[project.optional-dependencies].studio` (installed in both host `.venv` and `.venv-studio`).

## Unverified Items (Read-Only / Out of CPU Scope)
- Live Riva worker inference (`riva_local` checkpoint and worker python exist on disk; `device=cuda:0` configured, inference not run).
- Live Nemotron ASR (`asr_local`) and OpenAI-compatible text-model (`text_local`) inference.
- Real multi-track local media FFprobe (only single-track `.webm` files were present in inspected approved roots; multi-track index mapping verified via deterministic unit test).
- Interactive headless-browser DOM/JASSUB visual execution (TypeScript, Vite build, and Vitest unit suites verified; live Chromium click-through not run).
