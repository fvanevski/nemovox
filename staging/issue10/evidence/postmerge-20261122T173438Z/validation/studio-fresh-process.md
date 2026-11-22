# Phase 2/3: Fresh Production Studio Process & API Readback (postmerge-20261122T173438Z)

| Property | Pre-Promotion (`8fee1e1`) | Post-Promotion (`332a996`) |
|---|---|---|
| `MainPID` | `2922624` | `4180532` |
| `InvocationID` | `a863a06b44fe433493aea538ddbcb7f6` | `cd4b825f88744f3b8bd72153aa828aea` |
| `ActiveEnterTimestamp` | `Sat 2026-11-21 19:13:11 PST` | `Sun 2026-11-22 09:29:06 PST` |
| Loopback Bind | `127.0.0.1:8765` (single listener) | `127.0.0.1:8765` (single listener) |
| Served SPA Bundle | `assets/index-B12FMBjD.js` | `assets/index-62Ntthch.js` |
| `GET /api/health` | HTTP `200` | HTTP `200` |
| `GET /api/resource-status` | HTTP `404` | HTTP `200` (`asr: startable`, `translation_review: startable`) |
| `GET /api/translation-backends` | N/A | HTTP `200` (`riva.configured: true`, `device: "cuda:0"`) |

- Confirmed reading `GET /api/resource-status` left both `nemovox-asr.service` and `nemovox-text-review.service` in `ActiveState=inactive` (`MainPID=0`).
