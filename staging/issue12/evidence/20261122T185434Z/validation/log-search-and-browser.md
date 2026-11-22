
# Recent-Log Filter, Copy Visible vs Copy Full & Browser Acceptance

- **Candidate Studio:** `http://127.0.0.1:8769`
- **Browser:** `/usr/bin/chromium` (Playwright `sync_playwright`)
- **Uncaught `pageerror` count:** `0`
- **Console error count:** `0`

## Log Filter, Copy & Persistence (`000000000000000000000000000000cc`)
- Seeded `execution.log` (`161526` chars, >160 KB multiline UTF-8 `日本語 café 🎬` with 4200 early lines before the 100-line tail + 2 tagged tail lines `MATCH_TOKEN_UNIQUE_42`).
- Typed `MATCH_TOKEN_UNIQUE_42` into `Filter recent log lines`:
  - `<pre>` displayed only `['MATCH_TOKEN_UNIQUE_42 alpha café', 'MATCH_TOKEN_UNIQUE_42 beta 日本語']`.
  - `Copy visible` copied only those 2 filtered lines (`copied_filtered_exact=True`).
- Typed `NONEXISTENT_QUERY_XYZ`:
  - `<pre>` displayed `No matching recent log lines.`.
- Cleared filter and clicked `Copy full log`:
  - Clipboard received exact complete unfiltered `161526`-char log (`copied_full_exact_unfiltered=True`).
- Verified log panel persistence across `page.reload()` and `Browse all runs` drawer listing.
