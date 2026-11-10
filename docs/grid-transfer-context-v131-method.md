# GRID publisher URL correction before acquisition

The original v130 preparation stopped before freezing or downloading media: the
publisher's pinned API snapshot uses `/api/records/3625687/files/NAME/content`,
whereas the original planner admitted only the older `/api/files/` namespace.
No guarded stage or neural call occurred. Preserve the original code, tests,
method and private failure receipt unchanged.

This isolated v131 context validates each archive **and the corpus paper** against
the exact HTTPS host, record number, filename and content path. Reject credentials,
query strings, fragments, another record or another filename. Do not rewrite or
pretend that the publisher supplied legacy links. Bind the preserved failure and
original source hashes into the new freeze.

Use the same CC-BY-4.0 metadata snapshot, original sixteen-source hash ordering,
publisher sizes/checksums, eight two-archive batches, private Seagate storage and
[v130 acquisition procedure](grid-transfer-acquisition-v130-method.md). The fixed
future clip recipe, exclusion of noncommercial media, complete-body checks and
archive checks remain unchanged. Each full archive is downloaded once, one at a
time. Reuse the unchanged acquisition/guard code objects with new exclusive paths.

Test the new context, check the actual pinned metadata, freeze it and publish the
reviewed code/freeze manually before acquisition. Then run outside the sandbox:

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.grid_transfer_context_v131 run 0
```

Inspect the real profile/receipt before batches 1–7. Stop on a failure or denied
admission; preserve attempted stages and partial downloads, with no automatic
retry or overwrite. Data-only limits stay 900 seconds, 1.5 GiB sampled descendant
RSS, 3 GiB initial RAM/project/archive disk and 1.5 GiB running reserves. No model,
decoder, customer confidence change, Actions workflow, outreach or deployment is
part of this stage. Public aggregates/journal commitments contain no media,
individual source identities or individual media fingerprints.
