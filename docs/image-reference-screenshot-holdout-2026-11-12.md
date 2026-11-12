# Local changes in a photo inside a screenshot

The local earlier-copy tool now compares a native-scale photo inside a larger PNG screenshot. It shows the matched area in each file, the localized changes, and the exact earlier and submitted crops. The comparison PDF and JSON retain the same coordinates and visual evidence. Pixels outside the paired areas are unassessed. Local tone adjustments and interface overlays can also change pixels, so the comparison contributes zero weight to AI-origin confidence. It requires an independently supplied earlier photo. [App and offline review](image-reference-comparison-2026-11-11.md)

## Measurement

Exact three-pixel RGB patterns propose integer offsets between the earlier photo and the screenshot. The check requires a single placement with at least 75% exact RGB agreement inside the inset, continuous matching portions along all four sides, and corroboration from at least 12 cells across four rows, four columns and all quadrants. The cell patterns must be unique within the sampled earlier grid. The method reports connected groups of changed pixels within the proven rectangle and keeps both coordinate systems. A compact change may touch an edge. The check uses no learned weights.

The previous experimental inset rule often missed images with too few textured tiles or an edit that touched a side. The new method was frozen before scoring 18 other photo groups. All earlier RGB hashes, recorded source photo identities and contributor identities differ from both previously scored inset panels. These sources had appeared in other image studies. The screenshot deliveries are new. Chromium produced the original, edited and photo-free screenshots at one device pixel per source pixel. The harness checks that Chromium preserved the photo pixels exactly. Six additional controls per photo use deterministic local composites. [Prior inset experiment](image-reference-embedded-screenshot-2026-11-12.md) · [Frozen results](evaluation-panels/image-reference-screenshot-holdout-2026-11-12.json)

| Frozen outcome | Previous app | Previous inset candidate | New app check |
| --- | ---: | ---: | ---: |
| LaMa edit locations /12 | 0 | 8 | 12 |
| SDXL edit locations /3 | 0 | 0 | 2 |
| Kandinsky edit locations /3 | 0 | 3 | 3 |
| Clean photo placements /18 | 0 | 15 | 18 |
| Ordinary local tone differences located /18 | 0 | 15 | 18 |
| Incorrect accepted placements | 0 | 0 | 0 |
| Clean photo highlights /18 | 0 | 0 | 0 |
| Incompatible controls accepted /108 | 0 | 0 | 0 |

A useful location recovers at least half the true changed pixels with intersection-over-union of at least 0.30. The six incompatible control types are a photo-free screenshot, global tone adjustment, rescaling, unrelated photo, duplicate placement and JPEG delivery. Cases reuse the same 18 photos across control types. The crop layouts retain at least 80% of each known edit, so the test does not estimate recovery when part of the edited area was cropped away. The generated edits were feathered and preserve original RGB outside the annotated changes. Publisher contributions use CC BY 4.0, and the verified source photo records use CC BY 2.0. The local evaluation only reuses retained assets. [Dataset terms and attribution](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution) · [Source qualification](image-bg-real-transport-v225-result.md) · [Other editor asset and license audit](image-background-patch-transfer-2026-11-11.md)

The app's fallback covers one rectangular, native-scale photo inset in an upright, single-frame RGB PNG. The earlier photo must be at most 1,024 pixels on its longest side and the screenshot at most 1,280. The compared inset needs a minimum side of 160 pixels. A resized, recolored, compressed or ambiguous placement can remain unassessed. The check took a median of 30 ms and at most 48 ms on the local Mac. The existing earlier-copy feature is development-only and enabled through its existing setting.

## Reproduce

Use the retained rights-qualified sources, previous inset freezes and Chromium capture receipts. The media and per-case evidence stay local. From the repository root, set `alternate` to the qualified SDXL and Kandinsky holdout archive and `run` to a private run directory.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
study() { backend/.venv/bin/python -B -m scripts.image_reference_screenshot_holdout "$@"; }
study prepare "$run" --alternate "$alternate"
node backend/scripts/capture_image_crop_multiscale_transfer.cjs "$run/browser"
for stage in freeze run review; do
  study "$stage" "$run" --alternate "$alternate"
done
```

The retained run reproduced all 162 comparisons, including the exact geometry, selected-mask hashes and decisions. A local serving replay also checked three edited screenshots, three clean screenshots and three duplicate-placement controls. The edited examples reproduced two native crops each and 51,425 selected pixels across the three masks. Each crop, mask and overlay SHA-256 appeared in the rendered PDF. The clean and duplicate controls produced no evidence PNGs.

Use the existing earlier-copy export reviewer for an individual JSON download. It recomputes the two file hashes, native pixel and witness counts, difference components, source crops and transparent overlays before replaying the unique placement.

```sh
backend/.venv/bin/python -I backend/scripts/review_image_reference_oriented_resize_export.py \
  comparison.json earlier.png screenshot.png
```
