# Fresh image references: pixel and duplicate checks

This protocol follows the [completed 256-record metadata collection](image-fresh-attribution-v38-results.md).
It fetches every one of the 163 current photo/owner/licence matches without an
identified creator-key link to the 705-record source inventory. The 67 unresolved
metadata records and 26 known-creator matches remain in the earlier denominator.
No image is chosen using appearance, detector output or download success.

## Frozen acquisition

Use the original metadata order in six phases of 32, 32, 32, 32, 32 and three
references. Make one request for each exact provider-bound HTTPS static JPEG URL.
Allow no redirects, retries or replacement sources. Use 10/20-second connection
and read timeouts, a 5,000,000-byte body cap and at least 0.25 seconds between
request starts. Preserve errors, available partial bodies and failed cases.

Store encoded bodies by SHA-256 in a new private folder on the owner-authorized
Seagate drive. Verify its mounted, writable volume identity before preparing or
using the cache. Monitor internal and external free space. No existing files,
models, uploads, calibration or earlier cache attempts are overwritten.

Only JPEGs with native dimensions from 1 through 2,048 and EXIF orientation 1
or absent qualify. Unsupported orientation remains unresolved; no silent
rotation or resize is allowed. Hash the exact encoded bytes and native RGB,
retain geometry and reproduce the record from the saved file during readback.

## Fixed complete source graph

Join every qualified reference to all 705 existing records using the unmodified
v22 grouping implementation: same source ID, same canonical creator key, exact
RGB with matching dimensions, or informative 256-bit thumbnail dHash distance
at most eight with aspect ratio within 2%. Compute full transitive components.
Exclude any new-source component touching any prior inventory record, including
unscored or reserved records; do not check only previously consumed fit sources.

Report all 163 download outcomes, qualified references, failed reasons, prior
links and remaining candidate-only components. Keep source photos, identifiers,
profiles, exact pairings and thumbnail sketches private. Public output contains
methods, counts and hashes only.

## Execution evidence and limits

The exact source list, input/code hashes, package versions and resource rules
are committed by the [public freeze](evaluation-panels/image-fresh-source-cache-v39-freeze.json)
before the first image request. Each phase has a separate 900-second/6 GiB
sampled worker-tree RSS monitor, 1.5 GiB RAM/internal/external disk reserves,
initial 6 GiB available RAM and 3 GiB free on both disks. Preserve all observed
resource crossings, sampling errors and terminal statuses. No implicit retry,
overwrite or failed-to-passed relabelling is allowed.

The model-free preparation completed in 2.20 seconds with exit zero and no
monitor error or resource crossing; the largest observed worker RSS sample was
55,312,384 bytes, not a guaranteed peak. Fifty-three focused local tests pass,
including byte/geometry readback, unsupported orientation, exact-pixel links,
failed-request coverage and the earlier attribution checks. Ruff passes.

This does not establish camera authenticity, complete creator-alias resolution,
absence of transformed duplicates or independence from model pretraining.
Unknown creators remain unknown. No neural model, generator, fit, calibration
or detector evaluation runs here; no study role is assigned. Generator component
licences and bounded authored hardware/geometry probes remain separate admission
steps. The active training study and serving defaults are unchanged. No outside
person is contacted, no workflow runs, and no cloud resource is used.
