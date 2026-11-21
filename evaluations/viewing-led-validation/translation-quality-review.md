# Translation & Dialogue Quality Review

No new SwiftU GPU inference pass was needed: targeted overlay localization fixes in [fr_s04e01_validation.json](file:///workspace/nvidia-transcribe/configs/overlays/fr_s04e01_validation.json) and [zh_ep166_validation.json](file:///workspace/nvidia-transcribe/configs/overlays/zh_ep166_validation.json) resolved every conspicuous semantic error observed during clip playback while keeping pre-overlay text preserved in `pre_overlay_translated_text`.

## Confirmed Corrections (Audio/Hardsub Verified)

| Episode | Utterance | Before | After | Defect Type |
| :--- | :--- | :--- | :--- | :--- |
| FR | `utt-0031` | `...doesn't have a nat'cal like me...` | `...doesn't have a Naacal like me...` | Proper noun ASR corruption |
| FR | `utt-0079/81` | `Slowly, everything will` ... `fine.` (`7s` later) | `Easy now, everything will be fine.` | Split clause & delayed flash |
| FR | `utt-0082` | `ogre.` | *(suppressed)* | Nonexistent hallucination |
| FR | `utt-0117` | `It's going to rain!` | `It's raining rats!` | ASR misheard `Il pleut des rats!` |
| ZH | `utt-0019` | `...make one last stand. As the hidden hand...` | `—is not in my hands. Only Bai Xiaochun can still fight.` | 8s-early forward bleed / echo |
| ZH | `utt-0024` | `...will face a tragic end together.` | `...will face an unknown chapter together.` | ASR `为之偏张` vs `未知篇章` |
| ZH | `utt-0026` | `Ah, what difference is there between Xu Zheng and you?` | `What difference does the answer make to me or to you?` | ASR `于朕` garbled to `徐峥` |
| ZH | `utt-0102` | *(omitted)* | `...the Wildlands will unite as never before—ready for greater challenges.` | Fixed `满坊`/`祖国将上` ASR typos |
| ZH | `utt-0186` | `It's already settled.` | `This time, victory is mine!` | ASR `这一字已经定了` vs `这一次我赢定了` |
| ZH | `utt-0189` | `Underworld Emperor defers.` | `This is the sacred Underworld Emperor Palace.` | ASR `此乃冥皇宫圣地` garbled |
| ZH | `utt-0251` | `I, Bai Xiaochun, block you. Not ready yet.` | `I, Bai Xiaochun, will wait—but you're not ready yet.` | ASR `等着` vs `挡着` |
| ZH | `utt-0280` | `Gazing up, in reality...` | `Gazing toward you from afar, Master...` | ASR `遥望师尊` vs `仰望实则` |
| ZH | `utt-0282/83` | `Ah` / `Thank you.` | *(suppressed)* | Spurious closing split |

**Remaining minor items:** On EP166, `主宰大墓` (`Sovereign's Great Tomb`) at `06:11` renders `"Sovereign's great veil"` (`大幕`) and minor background crowd words enter `TopAlt` at `09:16–09:34`; both do not block plot comprehension.
