# Comparing similar patches to find small JPEG removals

Neither texture check added a correctly placed highlight on nine previously
unused photographs. The checks remain in evaluation. The local image
analysis continues to use its existing checks.

## Method

Compare a fitted local texture score with a second score that also uses
texture from similar parts of the submitted photo. The second version
compares the fine and medium detail of each 14-pixel tile with 24 tiles
with similar color, coarse detail, and gradient direction. It uses
neighbors at least five tiles away horizontally or vertically and measures how
far the tile's detail lies above or below the peers' median. The two
versions search identical rectangles and compare each rectangle with its
surroundings. They need no new pretrained model weights.

Fit separate parameters at JPEG qualities 75 and 90 using 36 older
object-removal pairs. Match each edited rectangle with the same area in
four ordinary controls and with a different area of the same size.
Twelve other photos fixed two operating points per method and quality.
The selection prioritized correct feathered locations. Neither method
added one, so the points with fewer extra ordinary alerts were selected.

## Held comparison

Nine additional BG-REAL photos supply 108 delivered JPEGs. Each photo has
a sharp object removal, the same removal with a feathered boundary, and
four controls covering the unchanged photo, local blur, local color
adjustment, and local copying. Six photos have a short side of at least
512 pixels. The other three are at least 384 pixels on that side.

The originals have individual CC BY 2.0 attribution. The publisher
releases the dataset contributions under CC BY 4.0. The publisher lists
these photos in its training split. They were held out from this project
comparison by source identity, contributor, file and pixel digests, and
near-duplicate fingerprints. All removals come from one editor. The
receipt gives each photo's attribution and pinned source path. [Publisher
terms](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)
· [Source receipt](evaluation-panels/image-removal-peer-sources-2026-11-16.json)

A correct highlight overlaps at least 30% of the union with the exact
changed-pixel region and recovers at least half of those pixels.
The baseline combines the existing JPEG coefficient check with the
regional check available at each quality.

| JPEG quality | Correct locations, baseline → combined | Ordinary files flagged, baseline → combined | Misplaced highlights added |
| --- | ---: | ---: | ---: |
| 75 | 0 → 0 of 18 | 4 → 8 of 36 | 2 |
| 90 | 0 → 0 of 18 | 0 → 4 of 36 | 2 |

The eight added control alerts come from one source photograph across both
qualities. Every edit has a rectangle in the search that meets the overlap
requirement. The highest-ranked rectangle from texture alone met that
requirement once at each quality, below its operating point. Comparing
similar tiles produced no correctly placed top rectangle at either
quality. The per-delivery receipt also preserves the wider settings and
separate size counts. [Held results](evaluation-panels/image-removal-peer-held-2026-11-16.json.gz)

## Replay

Run the recount with standard Python. Use
the pinned public-source recipe to reconstruct the JPEGs and exact masks
locally. The local replay rebuilt all 108 JPEGs and masks, then reproduced
the full held results byte for byte. The original input files and all
crops stay outside the repository.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export VERITAS_MODELS_DIR="$PWD/models" VERITAS_HF_HOME="$PWD/models/hf" VERITAS_OFFLINE=true
runner=backend/.venv/bin/python
panels=docs/evaluation-panels
module=scripts.image_removal_peer
python3 -B -S -m "$module" verify "$panels/image-removal-peer-held-2026-11-16.json.gz"
"$runner" -B -m scripts.image_cross_quality_sources restore \
  "$panels/image-removal-peer-sources-2026-11-16.json" "$run/rebuilt"
"$runner" -B -m "$module" fit "$retained" "$earlier_cross_quality" "$run/freeze.json.gz"
"$runner" -B -m scripts.image_region_rank_reference "$run/rebuilt" "$run/reference.jsonl"
"$runner" -B -m "$module" evaluate "$run/freeze.json.gz" "$run/rebuilt" \
  "$run/reference.jsonl" "$run/rebuilt-results.json.gz"
cmp "$run/rebuilt-results.json.gz" "$panels/image-removal-peer-held-2026-11-16.json.gz"
```

The fit uses the retained [JPEG ranking deliveries](image-region-ranking-2026-11-16.md#replay-and-compare-another-check)
and the [cross-quality confirmation archive](image-cross-quality-2026-11-16.md#exact-input-and-overlap-replay),
including their current-check measurements. Replaying regional inference uses the
previously verified checkpoint and the same runtime and JPEG encoder.
