# Translation Model Evaluation: Riva-Translate-4B vs SwiftU / NInfer

## Experimental Setup

Script: [compare_translations.py](file:///workspace/nvidia-transcribe/experiments/subtitle/compare_translations.py) (`20` curated French + Mandarin cases across 9 categories, with previous/next dialogue context attached and glossary hints supplied where relevant; outputs saved to `outputs/subtitle_runs/translation_comparison.json`).

- **Candidate A (`riva`)**: Local `/workspace/models/Riva-Translate-4B-Instruct-v2` (`torch.bfloat16` via isolated subprocess worker [_riva_translate_worker.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/_riva_translate_worker.py), `cuda:0`, peak VRAM **`8015.41 MiB`** FR / **`8017.8 MiB`** ZH). Strictly sentence-by-sentence (`fr-en`, `zh-en`), zero neighboring turn context or glossary prompting.
- **Candidate B (`swiftu`)**: Local `/workspace/ninfer/swiftu` served via `ninfer-serve` (`http://127.0.0.1:8001/v1`) with contextual `before`/`after` windows, glossary enforcement, and subtitle style/regrouping prompts.

## Quantitative Latency & Resource Summary

| Backend | FR Batch Wall (`10` cases) | ZH Batch Wall (`10` cases) | Combined Wall | VRAM Behavior | Contextual / Glossary Support |
| --- | --- | --- | --- | --- | --- |
| `riva` (4B BF16) | `16.66s` (`load=5.31s`, `infer=6.06s`) | `12.29s` (`load=2.84s`, `infer=4.77s`) | `28.95s` | ~`8.0 GiB` peak (`8017.8 MiB`) | Sentence-isolated; ignores surrounding dialogue & glossary |
| **`swiftu` (27B NVFP4)** | — | — | **`23.677s`** (`0.49s–2.24s`/case) | Co-located server (`~24 GiB`) | Full dialogue window + glossary + style control + safe regrouping |

## Side-by-Side Qualitative Findings (`20` Benchmark Cases)

Selected decisive cases across the 20-case bakeoff:

1. **`zh_04_mixed_script` (`cjk_latin_mixed`)** — `"Scotturb 403 路公共汽车定期发车前往辛特拉 (Sintra)，在罗卡角停靠。"`
   - `Riva`: `"Bus 403 runs regularly to Sintra, stopping at Roca Point."` *(dropped `"Scotturb"` brand name, mistranslated Cape Roca)*
   - `SwiftU`: **`"Scotturb Route 403 buses run regularly to Sintra, stopping at Cape Roca."`**
2. **`zh_06_expressive_colloquial` (`mandarin_expressive`)** — `"天哪，都火烧眉毛了，赶紧先走再说！"`
   - `Riva`: `"My goodness, everyone's eyebrows are on fire! Let's go now!"` *(literal idiom translation)*
   - `SwiftU`: **`"Oh my God, it's urgent! Let's just go!"`** *(natural English dialogue)*
3. **`fr_05_context_elliptical` (`context_dependent`)** — Prev: `"Combien de médaillons faut-il..."` Target: `"Il en faudrait deux, mais lui n'en a qu'un."`
   - `Riva`: `"He needs two, but he only has one."` *(mis-resolved impersonal `"Il en faudrait"` to `"He"`)*
   - `SwiftU`: **`"It would take two, but he only has one."`** *(correctly disambiguated from question context)*
4. **`zh_05_idiom_chengyu` (`mandarin_idiom`)** — `"真是画蛇添足，结果弄巧成拙，现在全乱套了。"`
   - `Riva`: `"It was really overkill, and the result was a mess. Now it's all messed up."`
   - `SwiftU`: **`"We really overdid it, and now everything's a mess."`** *(concise subtitle phrasing)*
5. **`fr_04_idiom_surprise` (`french_idiom_expressive`)** — `"Diablement efficace, n'est-il pas ? Pour sûr, c'est épatant !"`
   - `Riva`: `"It's devilishly effective, isn't it? It's certainly amazing!"`
   - `SwiftU`: `"Damn effective, isn't it? For sure, it's amazing!"` *(both valid; `natural` style favors conversational register)*
6. **Formal / Technical parity** (`fr_08_fleurs_00`, `fr_09_fleurs_03_mashhad`, `zh_02_fleurs_01_potassium`, `zh_03_fleurs_02_clearance`, `zh_10_science_fleurs`): both backends accurately preserved scientific terms (`calcium`, `potassium`, `dark matter`), Mashhad, dates (`August 2011` / `March 2017`), and numerals.

## Architecture Selection Decision

- **Default translator backend:** `swiftu` (`TranslationConfig.backend = "swiftu"`, `style = "natural"`), because it handles mixed-script terms (`Scotturb`, `Cape Roca`), elliptical pronouns (`Il en faudrait`), and idioms (`火烧眉毛`) noticeably better for subtitle viewing while also matching or beating cold subprocess Riva latency (`23.677s` vs `28.95s`).
- **Supported fallback backend:** `riva` remains implemented in [translation.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/translation.py) via `--translator riva` when a user prefers lightweight `~8 GiB` VRAM isolated sentence translation (`cuda` or `cpu`) without running an inference server.
