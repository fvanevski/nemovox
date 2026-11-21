# EP166 Untimed Dialogue Audit & Chronological Reconstruction

## 1. Root Cause of Cross-Chunk Misordering
In the EP166 baseline (`outputs/subtitle_runs/zh_full_episode_ep166/translated.contract.json`), **41** utterances lacked word/character CTC alignment (`source_start=None, source_end=None`) due to OOV/fast-recap acoustic tokens, scattered across **8 source chunks**:

| Chunk Index | Window `[start, end]` (s) | Untimed Count | Baseline Index Range | Reconstructed Chronological Position |
| :--- | :--- | :--- | :--- | :--- |
| 0 | `[0.0, 100.0]` | 4 | `0..3` | `utt-0000..utt-0003` |
| 1 | `[80.0, 190.0]` | 3 | `251..253` (wrongly pushed to end) | `utt-0006..utt-0008` |
| 3 | `[260.0, 370.0]` | 5 | `254..258` (wrongly pushed to end) | `utt-0023, 0026, 0029, 0032, 0035` |
| 4 | `[350.0, 460.0]` | 1 | `259` (wrongly pushed to end) | `utt-0040` |
| 7 | `[620.0, 730.0]` | 4 | `260..263` (wrongly pushed to end) | `utt-0102, 0103, 0107, 0112` |
| 10 | `[890.0, 1000.0]` | 12 | `264..275` (wrongly pushed to end) | `utt-0180, 0198..0208` |
| 11 | `[980.0, 1090.0]` | 11 | `276..283` + earlier misordered | `utt-0223..0226, 0229, 0234, 0244..0248` |
| 14 | `[1250.0, 1318.88]` | 1 | `284` | `utt-0284` |

**Why it happened:** `_compute_speaker_sort_keys` in `src/nvidia_transcribe/utterances.py` anchored untimed items using `float(cidx) * 1000.0`. For chunk `cidx=1`, `1000.0` exceeded nearly the entire timed timeline (`~1203.6s`), and for `cidx >= 3` (`3000.0..11000.0`), every untimed line from earlier scenes sorted strictly after the end of the episode.

## 2. Classification Summary (`audit_and_reconstruct_contract`)
Without fabricating `source_start`/`source_end`, `audit_and_reconstruct_contract` places all 41 items in chronological chunk order and classifies them into 4 buckets:

| Classification | Count | Interpretation |
| :--- | :--- | :--- |
| `plausible_independent_missing_timing` | **22** | Genuine coherent dialogue not covered by nearby timed lines |
| `uncertain` | **11** | Ultra-short interjections/fragments (`嗯。`, `也对。`, `哎呀。`, `天呐。`, `啊，`) |
| `probable_duplicate` | **7** | Chunk-overlap boundary echoes duplicating nearby timed lines (`sim >= 0.65`) |
| `likely_hallucination_or_loop` | **1** | Ending-credits repetitive loop (`utt-0284`, Chunk 14) |
| **Total** | **41** | Zero timestamps fabricated (`source_start=None, source_end=None` preserved) |

## 3. Complete 41-Utterance Audit Table

| Orig ID | New ID | Chunk | Window (s) | Spk | Classification | Dup Of (Sim) | Source Excerpt |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `utt-0000` | `utt-0000` | 0 | `[0.0, 100.0]` | `SPK_00` | `plausible_independent_missing_timing` | — | 两 个 月儿姑娘 说。 |
| `utt-0001` | `utt-0001` | 0 | `[0.0, 100.0]` | `SPK_01` | `uncertain` | — | 嗯。 |
| `utt-0002` | `utt-0002` | 0 | `[0.0, 100.0]` | `SPK_01` | `uncertain` | — | 嗯。 |
| `utt-0003` | `utt-0003` | 0 | `[0.0, 100.0]` | `SPK_01` | `uncertain` | — | 嗯。 |
| `utt-0198` | `utt-0006` | 1 | `[80.0, 190.0]` | `SPK_03` | `plausible_independent_missing_timing` | — | 我白小春人生选择，凡事发生，皆有利于。 |
| `utt-0199` | `utt-0007` | 1 | `[80.0, 190.0]` | `SPK_03` | `probable_duplicate` | `utt-0009` (`0.65`) | 全打青龙帮脚踩天宫会… |
| `utt-0200` | `utt-0008` | 1 | `[80.0, 190.0]` | `SPK_03` | `probable_duplicate` | `utt-0009` (`0.65`) | 更凭实力冲上了星空榜前列，获得去常… |
| `utt-0251` | `utt-0032` | 3 | `[260.0, 370.0]` | `SPK_04` | `plausible_independent_missing_timing` | — | 哦，我神明需要。 |
| `utt-0252` | `utt-0035` | 3 | `[260.0, 370.0]` | `SPK_04` | `probable_duplicate` | `utt-0029` (`0.65`) | 这次斩杀失败，才劳付一了 |
| `utt-0253` | `utt-0023` | 3 | `[260.0, 370.0]` | `SPK_05` | `plausible_independent_missing_timing` | — | 兆告天下，贤圣文治。 |
| `utt-0254` | `utt-0026` | 3 | `[260.0, 370.0]` | `SPK_06` | `plausible_independent_missing_timing` | — | 啊，徐峥与你又有什么区别？ |
| `utt-0255` | `utt-0029` | 3 | `[260.0, 370.0]` | `SPK_06` | `plausible_independent_missing_timing` | — | 不管继承人是谁，而等只是敬礼法子。 |
| `utt-0256` | `utt-0040` | 4 | `[350.0, 460.0]` | `SPK_07` | `plausible_independent_missing_timing` | — | 这呃明皇宫。 |
| `utt-0257` | `utt-0103` | 7 | `[620.0, 730.0]` | `SPK_14` | `plausible_independent_missing_timing` | — | 真会像传言一样，现任冥皇就是我家白浩… |
| `utt-0258` | `utt-0107` | 7 | `[620.0, 730.0]` | `SPK_14` | `plausible_independent_missing_timing` | — | 还好，真的是还好。 |
| `utt-0259` | `utt-0102` | 7 | `[620.0, 730.0]` | `SPK_15` | `plausible_independent_missing_timing` | — | 今夜之后，新任冥皇登基… |
| `utt-0260` | `utt-0112` | 7 | `[620.0, 730.0]` | `SPK_15` | `plausible_independent_missing_timing` | — | 一起，因缘，皆有天命。 |
| `utt-0261` | `utt-0198` | 10 | `[890.0, 1000.0]` | `SPK_13` | `uncertain` | — | 亏待。 |
| `utt-0262` | `utt-0180` | 10 | `[890.0, 1000.0]` | `SPK_13` | `plausible_independent_missing_timing` | — | 这回情仇是无人可及。 |
| `utt-0263` | `utt-0199` | 10 | `[890.0, 1000.0]` | `SPK_13` | `uncertain` | — | 也对。 |
| `utt-0264` | `utt-0200` | 10 | `[890.0, 1000.0]` | `SPK_13` | `uncertain` | — | 是神。 |
| `utt-0265` | `utt-0201` | 10 | `[890.0, 1000.0]` | `SPK_13` | `plausible_independent_missing_timing` | — | 正所谓知错能改，善莫大焉。 |
| `utt-0266` | `utt-0202` | 10 | `[890.0, 1000.0]` | `SPK_13` | `plausible_independent_missing_timing` | — | 嗨，亲爱的。 |
| `utt-0267` | `utt-0203` | 10 | `[890.0, 1000.0]` | `SPK_13` | `plausible_independent_missing_timing` | — | 白小纯。 |
| `utt-0268` | `utt-0204` | 10 | `[890.0, 1000.0]` | `SPK_13` | `uncertain` | — | 哎呀。 |
| `utt-0269` | `utt-0205` | 10 | `[890.0, 1000.0]` | `SPK_13` | `uncertain` | — | 天呐。 |
| `utt-0270` | `utt-0206` | 10 | `[890.0, 1000.0]` | `SPK_13` | `plausible_independent_missing_timing` | — | 一针四星，他弟亲亲抱助的大腿竟然是我的… |
| `utt-0271` | `utt-0207` | 10 | `[890.0, 1000.0]` | `SPK_13` | `plausible_independent_missing_timing` | — | 完了完了完了，大热大火。 |
| `utt-0272` | `utt-0208` | 10 | `[890.0, 1000.0]` | `SPK_13` | `plausible_independent_missing_timing` | — | 还不得白白浪费了这金金的气氛。 |
| `utt-0273` | `utt-0223` | 11 | `[980.0, 1090.0]` | `SPK_13` | `uncertain` | — | 天哪！ |
| `utt-0274` | `utt-0224` | 11 | `[980.0, 1090.0]` | `SPK_13` | `probable_duplicate` | `utt-0270` (`0.78`) | 一只死死趴地，勤勤抱住的大腿，竟然是… |
| `utt-0275` | `utt-0225` | 11 | `[980.0, 1090.0]` | `SPK_13` | `probable_duplicate` | `utt-0271` (`0.78`) | 完了完了完了！ |
| `utt-0276` | `utt-0226` | 11 | `[980.0, 1090.0]` | `SPK_13` | `probable_duplicate` | `utt-0272` (`0.78`) | 大人大不忿，还不得白白浪费了这千年的… |
| `utt-0277` | `utt-0234` | 11 | `[980.0, 1090.0]` | `SPK_13` | `plausible_independent_missing_timing` | — | 凭什么？ |
| `utt-0278` | `utt-0244` | 11 | `[980.0, 1090.0]` | `SPK_13` | `plausible_independent_missing_timing` | — | 为什么我恨你？ |
| `utt-0279` | `utt-0229` | 11 | `[980.0, 1090.0]` | `SPK_18` | `uncertain` | — | 嗯。 |
| `utt-0280` | `utt-0245` | 11 | `[980.0, 1090.0]` | `SPK_18` | `plausible_independent_missing_timing` | — | 主动你的计划而去，我不计前嫌送你一箭… |
| `utt-0281` | `utt-0246` | 11 | `[980.0, 1090.0]` | `SPK_18` | `plausible_independent_missing_timing` | — | 皇城满朝文武恨我入骨，我不惜暴露瑶琼… |
| `utt-0282` | `utt-0247` | 11 | `[980.0, 1090.0]` | `SPK_18` | `probable_duplicate` | `utt-0216` (`1.0`) | 而你呢？ |
| `utt-0283` | `utt-0248` | 11 | `[980.0, 1090.0]` | `SPK_18` | `uncertain` | — | 啊， |
| `utt-0284` | `utt-0284` | 14 | `[1250.0, 1318.88]` | `SPK_21` | `likely_hallucination_or_loop` | — | 嗯，对，就是，就是你，你，你，如果… |
