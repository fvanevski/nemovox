# Localhost HTTP & Browser Reserved-Filename Check (`bc5f067340d8bc84b7129bbdea7589ea401222c8`)

Executed against `.venv-studio/bin/python -m uvicorn nvidia_transcribe.studio_api:app --host 127.0.0.1 --port 8765 --workers 1` and Playwright Chromium (`/usr/bin/chromium --disable-gpu`).

## Synthetic Filenames Tested

- `output/Episode #1? 50% 日本語.en.srt` (`56` bytes, SHA-256 `4220b69dd79b4f8e9b7410890da65ca9ce662b202c1bd32d308ee2873a7614ca`)
- `output/Episode #1? 50% 日本語.en.ass` (`596` bytes, SHA-256 `6fc7c0d9d8c06994475a84f907cdd462433c30c492c780b5075692ba6d00ff5b`)

## `GET /api/jobs/{id}/artifacts` Output

```json
[
  {
    "name": "Episode #1? 50% 日本語.en.srt",
    "path": "output/Episode #1? 50% 日本語.en.srt",
    "url": "/api/jobs/44444444444444444444444444444444/artifact/output/Episode%20%231%3F%2050%25%20%E6%97%A5%E6%9C%AC%E8%AA%9E.en.srt"
  },
  {
    "name": "Episode #1? 50% 日本語.en.ass",
    "path": "output/Episode #1? 50% 日本語.en.ass",
    "url": "/api/jobs/44444444444444444444444444444444/artifact/output/Episode%20%231%3F%2050%25%20%E6%97%A5%E6%9C%AC%E8%AA%9E.en.ass"
  }
]
```

## Browser Request Paths, Downloads & ASS Loading

- **Playwright `expect_download()` clicks:**
  - `.srt` (`href="/api/jobs/44444444444444444444444444444444/artifact/output/Episode%20%231%3F%2050%25%20%E6%97%A5%E6%9C%AC%E8%AA%9E.en.srt"`): `56` bytes, SHA-256 `4220b69dd79b4f8e9b7410890da65ca9ce662b202c1bd32d308ee2873a7614ca` (exact byte match).
  - `.ass` (`href="/api/jobs/44444444444444444444444444444444/artifact/output/Episode%20%231%3F%2050%25%20%E6%97%A5%E6%9C%AC%E8%AA%9E.en.ass"`): `596` bytes, SHA-256 `6fc7c0d9d8c06994475a84f907cdd462433c30c492c780b5075692ba6d00ff5b` (exact byte match).
- **Browser HTTP log (`response` listener):**
  - `(200, '/api/jobs/44444444444444444444444444444444/artifacts')`
  - `(200, '/api/jobs/44444444444444444444444444444444/artifact/output/Episode%20%231%3F%2050%25%20%E6%97%A5%E6%9C%AC%E8%AA%9E.en.ass')`
- **JASSUB:** Activated `ASS · libass` overlay on synthetic WebM video with 1 canvas and `0` console/page errors.
