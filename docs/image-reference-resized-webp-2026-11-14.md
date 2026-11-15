# Earlier-copy evidence for a downsized WebP image

Chipmunk Verify can compare a proportionally downsized, lossy WebP image
with an earlier lossless copy. Open **Compare with an earlier copy** on a
retained image analysis. The browser, PDF and JSON show a brightness
difference mask in submitted coordinates and the corresponding native crop
from each file. The comparison is available in local development with
`VERITAS_IMAGE_REFERENCE_COMPARISON_ENABLED=true` and
`VERITAS_IMAGE_REFERENCE_LOSSY_WEBP=true`.

The earlier copy must be compatible opaque 8-bit RGB PNG or lossless WebP,
512–1,024 pixels on each side. The submitted file must be a simple opaque
VP8 WebP, at least 384 pixels per side, retaining 72–95% of the earlier
dimensions subject to image-dimension rounding. The comparison requires
matching framing and enough shared textured areas. It independently
projects the earlier copy with bicubic, bilinear and Lanczos filters. At
least two projections must support the framing. A highlighted pixel must
appear in at least two residual masks. The earlier crop bounds map the
submitted region outward to the earlier pixel grid. The crops always show
source pixels from their own files.
[Earlier same-size method](image-reference-lossy-webp-2026-11-14.md)
· [WebP container specification](https://developers.google.com/speed/webp/docs/riff_container)

## Held comparisons

The evaluation used 26 development and 24 separate held photo groups
from the existing WebP studies. Source hashes do not overlap between roles.
Photos from both banks had earlier project exposure. This test uses newly
prepared WebP deliveries at distinct resize ratios, qualities and encoder
effort levels. The area-resampled delivery uses a resize filter outside the
three filters measured by the comparison. Prior JPEG and WebP controls
were encoded and decoded before downsampling. The test also includes
untouched downsampling, mild local blur and tone, and pairs with translation,
cropping, stretching or unrelated content.
[Prior photo and edit rights](image-reference-lossy-webp-2026-11-14.md#held-comparison)
· [Matched-control methodology](https://arxiv.org/html/2607.26232v3)

The sharp and feathered edits come from known edited photos. The small
variants confine the largest local RGB change to a 52-by-52 square in the
earlier grid and apply a feathered border in one variant. Truth pixels are
the exact RGB changes after downsampling and before WebP encoding. A
useful location covers at least 50% of those pixels and reaches 0.30
intersection over union. The small variants test a selected change of this
size. They do not establish recall on faint or arbitrary small edits.

| Held delivery | Useful locations | Useful small edits | Unchanged or recompression-only flags | Wrong framing declined |
| --- | ---: | ---: | ---: | ---: |
| 75% bicubic, WebP quality 70 | 91/96 | 46/48 | 0/72 | 96/96 |
| 84% Lanczos, WebP quality 82 | 91/96 | 46/48 | 0/72 | 96/96 |
| 93% area, WebP quality 88 | 91/96 | 46/48 | 0/72 | 96/96 |

The previous same-size WebP comparison left all 936 held deliveries
unassessed. The new check located **273 of 288** known changes, including
**138 of 144** small changes. Fifteen other edit highlights fell below
the overlap requirement. No highlights appeared on the 144 mild blur or
tone controls. The deliveries share 24 underlying photos. Strong ordinary
edits can produce highlights. This comparison measures differences between
two files and carries zero AI-confidence weight.
[Development measurements](evaluation-panels/image-reference-resized-webp-development-2026-11-14.json)
· [Held measurements](evaluation-panels/image-reference-resized-webp-holdout-2026-11-14.json)

## Local app and independent replay

Three simulated prepaid analyses exercised one small edit, one previously
compressed control and one translated image. Jobs started only after
payment confirmation. The new route localized the small edit in a 576-by-384
WebP from a 768-by-512 source. Source replay confirmed both native crop
arrays and all 1,946 mask pixels, 1,708 of which overlapped the known edit.
The control had no selected region, and the translated pair was unassessed.
The local PDF and JSON kept the same source digests and crop coordinates.
The stored original analysis was preserved.
[App verification receipt](evaluation-panels/image-reference-resized-webp-app-2026-11-14.json)

Use the parent source layout from the
[same-size WebP replay](image-reference-lossy-webp-2026-11-14.md#local-replay).
The numeric recount uses standard-library Python. The native replay uses
the exact same resize and encoding versions and rebuilds every input and
mask. The runner uses libwebp 1.6.0. No source images are included in the
published measurements.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
dev=docs/evaluation-panels/image-reference-resized-webp-development-2026-11-14.json
held=docs/evaluation-panels/image-reference-resized-webp-holdout-2026-11-14.json
python3 -B -S -m scripts.image_reference_resized_webp_transfer review "$dev"
python3 -B -S -m scripts.image_reference_resized_webp_transfer review "$held"
backend/.venv/bin/python -B -m scripts.image_reference_resized_webp_transfer development "$run/dev.json" --development-sources "$photos"
backend/.venv/bin/python -B -m scripts.image_reference_resized_webp_transfer held "$run/held.json" --development-sources "$photos"
cmp "$dev" "$run/dev.json"
cmp "$held" "$run/held.json"
backend/.venv/bin/python -B -m scripts.image_reference_lossy_webp_app "$run/app" "$held" --resized --development-sources "$photos"
```
