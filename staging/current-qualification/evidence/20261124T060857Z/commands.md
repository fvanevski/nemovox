# Native execution summary

All commands ran from the isolated exact candidate through `rtk proxy`. Existing CLI and Studio interpreters were used with candidate `PYTHONPATH=src:.`. The assessment added a private wire receipt hook that recorded actual response bodies and image hashes without changing the request or result.

- Compile the adapter and both staged scripts. Exit 0. Verify imports in both interpreters. Exit 0 each.
- Run the five-file focused adapter and consumer suite. Exit 0, 92 passed. Run the full `-m 'not gpu'` suite in both interpreters. Exit 0 each, 397 passed each.
- Verify prior archive SHA-256, member checksums, manifest and original media hashes. Exit 0.
- Run `probe_quality.py --prior-root <verified prior> --prior-manifest <verified manifest> --output-dir <fresh inspection> --inspect-only`. Exit 0.
- Run `probe_quality.py --prior-root <verified prior> --prior-manifest <verified manifest> --fr-media <original> --zh-media <original> --output-dir <fresh live> --collect-only`. Exit 0. Existing default bindings and offline settings, no extra resource flags.
- Run `cached_asr_probe.py --checkpoint <verified local large-v3 snapshot> --inputs <live>/inputs.json --output <live>/independent-asr.json` with the existing faster-whisper interpreter, bytecode disabled and local-only CPU mode. Exit 0.
- Run `probe_quality.py --prior-root <verified prior> --prior-manifest <verified manifest> --output-dir <fresh assessment> --assess-collected <live> --independent-asr <live>/independent-asr.json`. Exit 0. The staged script supplied all native image blocks and obtained descriptions before the fused calls. No external visual descriptions were passed.
- Audit exact image receipts, response formats, explicit token budgets, and the per-service native journal. Exit 0. No application or staged script edits were required.
