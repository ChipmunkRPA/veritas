# Localizing changes in cropped WebP images

Chipmunk Verify can compare a cropped lossy WebP with an earlier lossless copy
of the same image. The comparison shows the area that was measured, the
locations of differences, and a crop from each file in its own pixel
coordinates. The browser, PDF and JSON use the same coordinates. Earlier-copy
comparison requires both files. Image differences carry zero weight in AI
confidence because ordinary editing can produce them.

## What the comparison measures

The earlier copy must be a supported opaque RGB PNG or lossless WebP with
sides of 512–1,024 pixels. The submitted file must be a simple opaque lossy
WebP with sides of 384–1,024 pixels. The crop retains at least 80% of each
dimension and removes at most 128 total pixels from either axis. Pixels outside
the matched crop are unassessed. A combined crop and resize is outside this
check's scope.

The app compares detail in nine distributed 48-pixel squares using normalized
correlation. At least five squares must independently support the same
integer-pixel offset, with no other offset supported by two or more squares.
The established WebP framing and luminance checks then confirm the placement
and select brightness differences after allowing for minor compression
variation. Weak, ambiguous or repeated detail can leave the pair unassessed.
[OpenCV template matching](https://docs.opencv.org/4.x/de/da9/tutorial_template_matching.html) ·
[Earlier same-size WebP comparison](image-reference-lossy-webp-2026-11-14.md)

## Local comparison

The held panel uses 24 photos separate from the 26-photo development bank.
These photos appeared in prior project studies. The asymmetric two-axis and
horizontal-only crop deliveries are new to this test. The four edit types are
sharp and feathered local replacements and their small variants. A useful
localization must recover the exact crop, cover at least half the authored
changed pixels, and reach at least 0.30 intersection over union. The two
deliveries per photo share a source. [Source selection and rights](image-reference-lossy-webp-2026-11-14.md#held-comparison)

| Held cases | Cases | Exact crop | Highlighted | Useful localization |
| --- | ---: | ---: | ---: | ---: |
| Staged local edits | 192 | 172 | 168 | 163 |
| Crop only or prior JPEG/WebP compression | 144 | 133 | 0 | — |
| Reshaped, mirrored or unrelated copies | 144 | 0 | 0 | — |

The prior comparison accepted none of the 192 staged edit deliveries. The
new check usefully located 84 of 96 small edits. It left 20 edited pairs and
11 ordinary crop or compression pairs unassessed. No case registered with a
wrong offset on this panel. Separate endpoint tests cover a change entirely
outside the crop and repeating texture with ambiguous placement. The held
receipt contains input, authored truth and output mask digests without source
media. [Held measurements](evaluation-panels/image-reference-cropped-webp-2026-11-14.json)

## Replay locally

Use the [parent source layout](image-reference-lossy-webp-2026-11-14.md#local-replay).
The native replay recreates the cropped deliveries and mask hashes. The app
exercise uses simulated checkout for an edited file, a recompressed control,
and an incompatible pair. It checks the PDF and replays the JSON against the
submitted and earlier source pixels. [App verification receipt](evaluation-panels/image-reference-cropped-webp-app-2026-11-14.json)

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
held=docs/evaluation-panels/image-reference-cropped-webp-2026-11-14.json
python3 -B -S -m scripts.image_reference_cropped_webp_transfer review "$held"
backend/.venv/bin/python -B -m scripts.image_reference_cropped_webp_transfer held "$run/crops.json" --development-sources "$photos"
cmp "$held" "$run/crops.json"
backend/.venv/bin/python -B -m scripts.image_reference_lossy_webp_app "$run/crop-app" "$held" --cropped --development-sources "$photos"
```
