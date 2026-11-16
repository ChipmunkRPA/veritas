# Small rotated and resized copies in a single WebP

Chipmunk Verify can highlight two areas in one lossy WebP that share detail
after rotation or resizing. The extra search runs when the standard geometry
check finds no supported pair. It aligns two candidate areas and requires
agreement in color and fine detail. The app shows both original crops, their
coordinates and the selected pixels. Either area could have been copied.
Repeated objects and ordinary edits can also produce a match. This check
requires no additional weights and contributes zero weight to the AI assessment.
[Underlying method](image-small-affine-jpeg-2026-11-16.md)

## Local comparison

The first panel converts the held quality-90 JPEGs from the earlier study to
WebP. It contains 18 CC0 camera photos and nine photos with recurring natural
or architectural patterns. A second panel uses ordinary JPEGs from ten other
CC0 camera groups. It applies the edits after decoding and resizing those
JPEGs so the staged edits undergo only the final WebP encode. The two panels
have disjoint camera IDs. The pictures appeared in earlier studies for other
deliveries or checks. [First-panel source terms](image-small-affine-jpeg-2026-11-16.md#method-and-source-panel)
· [Separate camera source terms](image-jpeg-native-rotation-2026-11-15.md)

Each camera photo contributes three staged copies and five ordinary controls.
Two encodes of each file use WebP quality 75 and 90 with effort 4. Copy paths
must locate both areas with at least 30% intersection over union and at least
50% recovery of each known area. The standard, expanded luminance-only, and
expanded color-and-detail comparisons use the same delivered pixels. The extra
search uses the settings from the JPEG study without fitting to WebP.

| Panel and WebP quality | Standard useful | Extra luminance-only useful | Extra color-and-detail useful | Ordinary flags, standard / luminance-only / color-and-detail |
| --- | ---: | ---: | ---: | --- |
| Converted JPEGs · 75 | 5/54 | 11/54 | 8/54 | 0/99 · 0/99 · 0/99 |
| Converted JPEGs · 90 | 6/54 | 17/54 | 16/54 | 0/99 · 0/99 · 0/99 |
| Separate cameras · 75 | 2/30 | 6/30 | 8/30 | 0/50 · 4/50 · 0/50 |
| Separate cameras · 90 | 3/30 | 8/30 | 9/30 | 0/50 · 5/50 · 0/50 |
| Combined | 16/168 | 42/168 | 41/168 | 0/298 · 9/298 · 0/298 |

The color-and-detail check adds 25 useful locations across 12 camera groups.
It also adds 19 partial matches that overlap both known areas below the
overlap requirement, and one highlight that fails to overlap both areas.
Every input was assessed. The 466 deliveries share 37 photo groups. The
results describe geometric copies and these encoding recipes. Coverage of
generative inpainting and other WebP encoders remains unevaluated here.
[Converted JPEG panel](evaluation-panels/image-small-affine-webp-transfer-2026-11-16.json.gz)
· [Separate cameras](evaluation-panels/image-small-affine-webp-cameras-2026-11-16.json.gz)

## Local app

The existing `VERITAS_IMAGE_AFFINE_COPY_REVIEW` and
`VERITAS_IMAGE_SMALL_AFFINE_COPY_REVIEW` development settings enable the check.
It covers simple opaque VP8 WebP stills with a long side of 768–1,024 pixels
and a short side of at least 384 pixels. The original filename plays no role in
eligibility. [WebP container specification](https://developers.google.com/speed/webp/docs/riff_container)

Eight simulated USD 9.99 checkouts tested two staged edits and their matched
originals with the extra search off and on. No job queued before payment.
The ordinary view, Evidence API, JSON export, and Standard PDF agreed on both
source crops and masks for each edit. The signed packages verified against
the original WebP files. The two ordinary controls produced no pair. The 50
focused image regressions passed with test defaults. Two existing offline
dependency assertions outside this change still fail for the geometric image
and video articulation bindings. [Local app receipt](evaluation-panels/image-small-affine-webp-app-2026-11-16.json)

## Replay

Prepare `$jpeg` from the earlier [small JPEG copy study](image-small-affine-jpeg-2026-11-16.md#reproduce).
Use the `held` directory from the [right-angle JPEG study](image-jpeg-native-rotation-2026-11-15.md#reproduce)
as `$camera`. The freeze records incoming JPEG and native truth hashes.
The numeric reader uses standard-library Python and the published receipts.
The native comparison needs Pillow with libwebp, NumPy and OpenCV. Both
panels reproduced byte for byte with a different number of workers.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
module=scripts.image_small_affine_webp
published=docs/evaluation-panels
python3 -B -S -m "$module" review "$published/image-small-affine-webp-transfer-2026-11-16.json.gz"
python3 -B -S -m "$module" review "$published/image-small-affine-webp-cameras-2026-11-16.json.gz"
backend/.venv/bin/python -B -m "$module" prepare "$jpeg" "$run/inputs"
backend/.venv/bin/python -B -m "$module" capture "$run/inputs" "$run/deliveries.json.gz"
backend/.venv/bin/python -B -m "$module" prepare-scenes "$camera" "$run/deliveries.json.gz" "$run/scenes"
backend/.venv/bin/python -B -m "$module" capture "$run/scenes" "$run/cameras.json.gz"
backend/.venv/bin/python -B -m scripts.image_small_affine_jpeg_app \
  "$run/app" "$run/deliveries.json.gz" "$run/inputs" --webp --browser
```
