# Source Refinement Evaluation (SwiftU / NInfer)

## Runtime Discovery: NInfer Thinking Budget vs `max_tokens`

During live execution against `ninfer-serve` (`Qwen3.8-27B-swift15-uncensored` at `http://127.0.0.1:8001/v1` started with `--thinking-budget 16384 --preserve-thinking`), requests without `reasoning_effort` spent the entire `max_tokens=4096` budget on internal `<think>` reasoning, truncating JSON mid-string (`finish_reason=length`) and triggering ~14.5s halving retries per batch.

Inspection of `/workspace/ninfer/src/serve/openai_chat_request.cpp` confirmed native top-level `reasoning_effort` support (`"none" | "minimal" | "low" | ...`). Testing a live 6-utterance French refinement batch on `127.0.0.1:8001`:

| `reasoning_effort` | Latency (s) | `finish_reason` | Parse status | Reasoning tokens | Total tokens |
| --- | --- | --- | --- | --- | --- |
| `"low"` | **4.62s** | `stop` | `ok` (6/6) | 1,038 | 2,152 |
| `"minimal"` | 24.22s | `stop` | `ok` (6/6) | 5,603 | 6,883 |

`RefinementConfig` and `TranslationConfig` therefore default to `reasoning_effort="low"` and `max_tokens=8192` via `OpenAIChatTextClient`, giving ~5x faster completion without JSON truncation.

## 4-Profile Comparison (French Episode Excerpts `00:00–06:15`, 68 Utterances)

Run via [run_live_subtitles.py](file:///workspace/nvidia-transcribe/experiments/subtitle/run_live_subtitles.py) (`--profile-comparison`, glossary [mcog_fr.yaml](file:///workspace/nvidia-transcribe/configs/glossaries/mcog_fr.yaml), `batch_size=6`):

| Profile | Changed / 68 | Rejected by screen | Batch errors | API calls | Elapsed (s) | Behavioral takeaway |
| --- | --- | --- | --- | --- | --- | --- |
| `off` | 0 | — | — | — | 0.000 | Raw ASR preserved (`convoi`, `la Gia`, `pour pourquoi`) |
| `conservative` | 13 | 0 | 0 | 12 | 44.618 | Capitalization & glossary spelling only (`Cités d'Or`); missed `convoi -> Condor` & `la Gia -> Zia` |
| **`balanced` (default)** | **10** | **0** | **0** | **12** | **46.195** | Fixed contextual homophones (`convoi -> Condor`, `la Gia -> la Zia`), stutters (`Tu tu tu tu sens -> Tu sens`), and capitalization without meaning drift |
| `expressive` | 29 | 0 | 0 | 12 | 62.165 | Fixed `à rien à manger -> sans rien à manger`, but drifted meaning on `la Gia -> la fille` (`utt-0033`) and churned punctuation |

### Representative Profile Diffs (`outputs/subtitle_runs/profile_comparison/profile_comparison.json`)

- **`utt-0001` (`19.08s–22.413s`)**:
  - `raw`: `"convoi, tu recherches les cités d'or."`
  - `conservative`: `"convoi, tu recherches les Cités d'Or."`
  - `balanced`: `"Condor, tu recherches les cités d'or."`
- **`utt-0033` (`239.32s–242.435s`)**:
  - `raw`: `"Il doit pas il doit penser à Isabella, la Gia."`
  - `balanced`: `"Il doit pas il doit penser à Isabella, la Zia."`
  - `expressive`: `"Il doit pas… il doit penser à Isabella, la fille."` *(hallucinated generic noun instead of character name)*
- **`utt-0050` (`289.24s–290.104s`)**:
  - `raw`: `"Tu tu tu tu sens"` -> `balanced`: `"Tu sens"`

## Full French Episode (`221` Utterances) & Mandarin Samples (Default `balanced`)

Across all `221` utterances of *Les mystérieuses cités d'or S04E01*, `balanced` applied **48** contextual edits (`0` screen rejections):
- Fixed recurring character & proper-noun ASR errors: `"convoi"` -> `"Condor"` (`utt-0001`), `"Lagira"` -> `"Laguerra"` (`utt-0018`), `"la Gia"` -> `"Zia"` (`utt-0033`), `"Dia!"` -> `"Zia!"` (`utt-0181`, `utt-0183`), `"Tado!"` / `"Bonjour, Tas!"` -> `"Tao!"` (`utt-0138`, `utt-0141`, `utt-0170`), `"Oranaori"` -> `"Rana'Ori"` (`utt-0147`), `"Pikachu"` -> `"Pichu"` (`utt-0212`, `utt-0213`), `"la Libella"` / `"Gebre Mesquel Libella"` -> `"Lalibela"` / `"Gebre Mesqel Lalibela"` (`utt-0068`, `utt-0200`, `utt-0207`, `utt-0218`).
- Fixed French grammar/stutter artifacts: `"Qu'elles feutré vos vos vos portes!"` -> `"Qu'elles soient feutrées, vos portes!"` (`utt-0025`), `"Nous va s'écrouler"` -> `"Ça va s'écrouler"` (`utt-0166`), `"Ça se trouve par un"` -> `"Ça se trouve partout"` (`utt-0171`).
- **Observed glossary-gap limitation (`utt-0152`)**: Because `"Naacal"` was absent from [mcog_fr.yaml](file:///workspace/nvidia-transcribe/configs/glossaries/mcog_fr.yaml), ASR mangling `"Un Akâa"` was over-normalized to the nearest glossary name `"Un Tao"`. Keeping character/clan terms like `Naacal` in series glossaries avoids nearest-neighbor substitution.

Across the four repaired Mandarin fixtures (with compact glossary `[Scotturb, Sintra, Cape Roca, calcium, potassium]`):
- `zh_fleurs_01_2006`: `0` edits (`1.019s`), raw text already exact (`钙、钾等元素属于金属，银和金等元素当然也是金属。`).
- `zh_fleurs_05_1805`: `1` edit (`1.701s`), repaired `"Scouter四零三路公交汽车定期发车前往新特拉 在罗卡角停靠"` -> `"Scotturb四零三路公交汽车定期发车前往辛特拉 在罗卡角停靠"`.
- `zh_2spk_overlap`: `2` edits (`5.735s`), repaired `"斯科特四零三路公交车车 去 定期 发车前往 洛杉矶 在 洛卡角 停靠"` -> `"斯科特四零三路公交车，定期发车前往辛特拉，在洛卡角停靠。"` (recovering Sintra from `"洛杉矶"`!).
- `zh_multichunk_synthetic_qwen`: `3` edits (`4.949s`), including homophone `"桥下垂直近空十五米"` -> `"桥下垂直净空十五米"` (`utt-0002`) and stutter `"位于大气 大气条件优良"` -> `"位于大气条件优良"` (`utt-0005`).
