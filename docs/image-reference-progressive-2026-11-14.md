# Exact earlier-copy comparison for progressive JPEGs

The local **Compare with an earlier copy** check now accepts progressive
encoding on either side of the exact-block JPEG90 fallback. The two copies
must have matching dimensions of 512 to 768 pixels per side and the same
standard quality-90 quantization tables and 4:2:0 sampling. Both must be upright
RGB and have no embedded color profile. A progressive file must have a
complete single frame and fixed quantization tables. The existing threshold
for matching decoded blocks and the difference calculation stayed fixed.
The result identifies changes between the supplied copies and carries zero
AI-authorship weight. [Earlier exact-block comparison](image-reference-same-jpeg-confirmation-2026-11-13.md)

## Local comparison

Twelve previously acquired LaMa object-removal photographs came from two
qualified BG-REAL source rosters. The first six qualifying photos per roster
were selected by original file hash. Photo, photographer and original byte
identities were distinct across this panel. Each group supplied sharp and
feathered removals, matched local blur, tone and copy edits, and appropriate
unchanged, globally processed and incompatible pairs. Original photos retain
their CC BY 2.0 credits. The publisher documents the separate CC BY 4.0
terms for its edit contributions. The photographs had previous project
exposure. [Primary source qualification](image-bg-real-transport-v225-result.md) · [Backup source qualification](image-bg-real-backup-intake-v262-result.md) · [Publisher terms](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)

For each of the 132 baseline pairs, losslessly transcode the earlier copy,
the submitted copy, and both. Across all 396 pairs with progressive
encoding, decoded RGB and compression profiles matched the baseline.
The route and exact-block evidence returned the same results and the same
mask, crop and overlay bytes, with each file's own source hash.
The separate JPEG tone and structure fallback was disabled in the carrier
comparison to isolate the exact-block path. The endpoint regression tests
also ran with the ordinary earlier fallbacks enabled. [Lossless JPEG transcoding](https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/doc/usage.txt)

| Outcome | Previous progressive route | With exact-block fallback |
| --- | ---: | ---: |
| Known object removals localized | 20/24 | 24/24 |
| Deliberately changed local controls highlighted | 28/36 | 34/36 |
| Unchanged or globally processed pairs highlighted | 4/36 | 4/36 |
| Incompatible pairs assessed | 0/36 | 0/36 |

A useful location recovers at least half of the known pre-encode changed
pixels and reaches 0.30 intersection over union. The four additions cover two
photos, one from each roster. The six additional local controls each
overlap the deliberately applied change. The test measures comparison
coverage for a supported JPEG encoding and uses an earlier copy for every
case. It supplies no new accuracy estimate for standalone image screening.

## Replay

The authenticated local comparison tests cover all four encoding pairings,
crop and mask parity in PDF and JSON, original-source replay, invalid
progressive transport, and rollback. To recount the
[published measurements](evaluation-panels/image-reference-progressive-2026-11-14.json),
use the standard-library reviewer.

```sh
PYTHONPATH=backend python3.12 -B -S -m scripts.image_reference_progressive_transfer review \
  docs/evaluation-panels/image-reference-progressive-2026-11-14.json
```

Use `run` with a new output path and the two verified `--rosters`
directories in the same primary, backup order to regenerate the measured
pairs. The native replay requires the local backend environment, retained
source bytes, and `jpegtran`. The public receipt contains hashes and scalar
measurements.
