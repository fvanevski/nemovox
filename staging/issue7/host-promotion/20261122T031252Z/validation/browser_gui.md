# Fresh Browser GUI Verification (`20261122T031252Z`)

- **Browser:** `/usr/bin/chromium` launched via Playwright headless (`1440x960`) against live `http://127.0.0.1:8765/` (`MainPID=2922624`).
- **Console / Page Errors:** `0` console errors, `0` uncaught page errors.
- **Serving & Navigation:** Loaded production SPA (`assets/index-B12FMBjD.js`, `assets/index-mYr5Cx9X.css`, JASSUB worker/WASM/font assets all HTTP 200). Navigated across `Media & task`, `Configure`, and `Process & preview`; `page.reload()` preserved `Recent runs` and the 2 rendered cues.
- **Configuration Defaults & Parity:**
  - `Spoken language`=`auto`, `Output language`=`en`, `Transcription mode`=`Speaker-attributed (recommended)`, `Primary translation engine`=`Riva Translate · dedicated translation model` (`Dedicated Riva translation worker is configured.`), `English translation review`=`Balanced (recommended)`, `Localization style`=`Natural dialogue`, `Character / terminology glossary`=`No glossary (default)`, `Optional extra condensation pass (advanced)`=`unchecked` (`condense_high_cps=false`), zero source-correction controls.
  - Selecting `Spoken language = en` updated the card to `English source · translation bypassed`.
- **Interactive Combobox DOM (`ExplainedSelect`):**
  - `Transcription mode` trigger exposes `role="combobox"`, `aria-haspopup="listbox"`, `aria-controls="_r_4_"`.
  - `ArrowDown` opened listbox (`aria-expanded="true"`), listed 4 ASR options, set `aria-activedescendant="_r_4_-opt-1"`, and updated `<p aria-live="polite">` to `"Overlapping speakers: ..."`.
  - `End` moved `aria-activedescendant` to `_r_4_-opt-3` (`"Rich transcription: ..."`).
  - `Home` moved `aria-activedescendant` to `_r_4_-opt-0` (`"Speaker-attributed (recommended): ..."`).
  - `ArrowDown` x2 + `Enter` committed `Verbatim transcript`, closed listbox (`aria-expanded="false"`), cleared `aria-activedescendant`.
  - `ArrowDown` + `Escape` closed without changing committed value.
  - Pointer click on option 0 committed `Speaker-attributed (recommended)` and restored DOM focus (`document.activeElement === combobox` evaluated `true`).
  - **Screen-reader audio:** `UNVERIFIED` (DOM ARIA/live-region state verified in Chromium; no audio TTS engine attached).
- **Video & JASSUB ASS Preview:**
  - `<video>` loaded synthetic 4.008s 640x360 WebM (`readyState=4`, `error=null`), seek/cue-click to cue 2 highlighted cue 2 (`data-active="true"`, `00:02.64`), and JASSUB/libass rendered live onto `.preview-pane canvas` with pill `ASS · libass` (visual screenshot confirmed synthetic testsrc2 video + rendered `"Synthetic cue two with second speaker."` subtitle).
