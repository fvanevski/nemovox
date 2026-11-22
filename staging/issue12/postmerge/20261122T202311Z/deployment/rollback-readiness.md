
# Rollback Readiness Receipt

- **Rollback invoked:** `no` (promotion & live acceptance succeeded cleanly)
- **Previous live commit/tree:** `332a99697edcc0b72fcc842168d5b1f57782e203` / `6f1b41c26f8439d9eb1c5a4aff11b64e4c40afaa`
- **Private host backup path (retained on host, NOT pushed to GitHub):** `/workspace/nemovox-issue12-rollback-20261122T201716Z`
  - Contains `web-dist-backup/` (pre-stop copy of old `web/dist`), `web-dist-live-prev/` (atomic rename swap source), `pre_hashes.sha256`, and `pre_jobs.sha256`.
  - Old `web/dist/index.html` sha256 verified matching backup (`d745cf10189c6c67f7bde1aef66e72360f5ecc4df8d9c71796aeae2eafeca38b`).
- **Rollback procedure if ever needed:** `systemctl --user stop nemovox-studio.service`, detach checkout to `332a99697edcc0b72fcc842168d5b1f57782e203`, restore `web-dist-live-prev` back to `web/dist`, `systemctl --user start nemovox-studio.service`.
