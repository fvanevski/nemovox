
# Qualification Findings

- **Blocking findings:** `none`
- **Source repairs required:** `none` (all checks passed against unmodified Central commit `99a3abce9cf8e8e5d894892f91a899c46e71a508`)
- **Host environment notes:**
  - Loopback port `127.0.0.1:8766` was already bound on the host by an unrelated MCP server process (PID `12704`), so per `HOST_VERIFY.md` (`choose another free loopback port if needed`) isolated candidate Studio verification ran on `127.0.0.1:8769` (and `127.0.0.1:8770` for stubbed remote/mock runs).
  - Observed one real UX nuance (non-blocking, works truthfully by design): when submitting an existing `recording.contract.json` with task `subtitles` while leaving the default `Source transcript (.txt)` export checkbox enabled, `asr/transcript.txt` is not generated (since ASR is skipped; `output/raw_source_transcript.txt` is generated instead), so `export` honestly reports `partial` (`source_transcript: failed`, `final_english`/`srt`: `saved`) while `job.state` remains `complete`.
