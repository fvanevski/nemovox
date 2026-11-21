# Local-agent evidence staging

Evidence must be published on dedicated GitHub evidence branches, not in main.

Recommended GUI handback layout:

~~~text
staging/agent-evidence/gui-foundation-20261121/
    gui-foundation-evidence.tar.gz
    manifest.json
    HANDback.md
    test-results.md
    browser-review.md
~~~

This repository is PUBLIC. Publish synthetic-only screenshots/fixtures, source diffs and sanitized test evidence. Never publish copyrighted media, derived TV/anime transcripts, credentials, model weights or private host logs. Keep individual archives small, ideally below 20-25 MiB. Force-add only the approved tar.gz with git add -f because the root .gitignore excludes archives.

Return full branch SHAs, branch links, artifact path, byte size and SHA256 for central independent review. Do not merge evidence archives to main.
