# Translation Review & Polarity Guard Qualification (`81704133`)

- **CPU Polarity Unit Check (`PASS`):** Verified `_unsupported_polarity_inversion` vetoes French+Riva negation drop (preserving Riva English MT cleanly) while allowing equivalent negative phrasings and non-negative corrections.
- **Full-Episode `pause_aware` Run (`169` Utterances -> Riva `fr-en` `60.6s`, `0` Errors -> Qwen `balanced`/`natural` `29` API Calls, `144.3s`, `0` Errors):**
  - **`49/169` (`29.0%`) changed**, **`120/169` (`71.0%`) unchanged**, **`rejected_polarity=0`**, cache hits `0` (clean fresh evaluation).
  - **Why `rejected_polarity=0` at `S02` (`u050` `[337.37–341.69]`):** Under `c08` (`[303.19..357.19]`), ASR misheard the unfamiliar `naacal` negation sentence completely into an affirmative proper-name phrase, so neither French source nor Riva MT contained the `S02` negation pattern on that chunk (`rejected_polarity` guard had `0` false vetoes elsewhere).
  - **Verified Helpful Qwen Repairs in Full Run:** `u037` (*nave -> ship* false-friend nautical fix), `u039` (*like him -> like her*), `u031` (omitted pronoun fix), `u154` (omitted numeral *eleven* restored).
  - **Remaining Review Edge Case:** `u026` truncated trailing object pronoun (`... his crew` -> `... his ship and`).
