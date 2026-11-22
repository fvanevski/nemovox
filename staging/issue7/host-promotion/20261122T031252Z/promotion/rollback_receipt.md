# Rollback Receipt (`20261122T031252Z`)

## Availability & Preimage Backup (HOST_ATTESTED, Retained Privately on Host)
- **Rollback status:** `available` (not invoked; promotion and smoke succeeded).
- **Private preimage backup directory:** `<REPO_ROOT>/outputs/handbacks/host-promo-private-20261122T031252Z/preimage/`
  - `nemovox-studio.service.bak`: exact copy of `systemd --user` unit file prior to restart.
  - `web_dist_pre.tar.gz` + `web_dist_pre.sha256`: complete backup of prior `web/dist` (`index-Bwa8rYL-.js`, SHA256 `4641218b4e9cb941bf966ac55b17563c2cf5626e43e03872acc2ecdf4dbce542`).
  - `systemd_pre.txt`, `health_pre.txt`, `index_pre.html`, `backends_pre.json`: pre-promotion live process and HTTP observations.

## Reversible Restore Procedure
1. Verify clean checkout and fast-forward origin: `git -C <REPO_ROOT> reset --hard 8205d7783230272c39a669c3f51d89bca50669e6`.
2. Restore `web/dist` from `web_dist_pre.tar.gz`.
3. `systemctl --user restart nemovox-studio.service` and confirm `/api/health` and `index-Bwa8rYL-.js` readback.
