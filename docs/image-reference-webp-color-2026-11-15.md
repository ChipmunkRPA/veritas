# Local color differences in lossy WebP

The earlier-copy comparison now has an optional check for color changes that leave
brightness nearly unchanged. It reuses the established framing for a supported
WebP upload and a lossless earlier copy. This includes same-framing images,
unscaled crops, proportional downsizing, and crops that were also downsized.
Every highlighted area includes the exact decoded crop from each file and an
overlay in the submitted image's coordinates.

The check smooths fine compression variation and compares two color channels
after allowing for an adjustment across the whole image. It fits the adjustment
on one sample grid and tests agreement on a different grid. For resized images,
at least two supported resize filters must select a pixel. The color areas are
separate from the retained brightness findings. This comparison measures a
difference between two sources. Ordinary recoloring can produce the same
evidence, and the comparison carries zero weight in the AI assessment.
[WebP format](https://developers.google.com/speed/webp/docs/riff_container) ·
[Color differences in image forensics](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Media%20Forensics/Stanton_Detecting_Image_Forgery_Based_On_Color_Phenomenology_CVPRW_2019_paper.pdf)

## Local evaluation

Eight of the attributed BG-REAL photos were used for development. Eighteen
source groups were held out from developing the color rule. The same rules ran
on ten other CC0 camera photos prepared for the native JPEG study. The camera
inputs use the earlier JPEG deliveries decoded into lossless PNG. These source
sets had been used for different image checks. There are two WebP deliveries
per source and multiple conditions per delivery. Files from the same source
share content. The counts describe this panel.
[Source attribution and native replay](image-edge-context-new-sources-2026-11-14.md#photo-qualification) ·
[CC0 camera intake](image-jpeg-native-rotation-2026-11-15.md)

A useful highlight covers at least half of the exact change and reaches 0.30
intersection over union. The held count includes files whose framing check
abstained. The supplied inputs include the original published edits, color-only
versions of those edits, and authored local recolors. The whole-image control
applies tint, hue or saturation to the entire image.

| Held condition | Files | Brightness alone | With color check |
| --- | ---: | ---: | ---: |
| Sharp and feathered local recolors, useful highlight | 72 | 0 | 68 |
| Recolor and separate brightness edit, useful combined mask | 36 | 0 | 32 |
| Published edit with its brightness removed, useful highlight | 36 | 0 | 15 |
| Published edit as delivered, useful highlight | 36 | 33 | 33 |
| Unchanged or intermediate JPEG/WebP, any highlight | 108 | 0 | 0 |
| Whole-image color change, any highlight | 108 | 0 | 11 |
| Translated or unrelated file, comparison abstained | 72 | 72 | 72 |

The color region by itself reached the overlap target on 33 of 36 files with a
separate brightness edit. The 108 unchanged or recompressed controls include
six framing abstentions. The whole-image adjustment residual produced three
hue, six saturation, and two tint flags. The ten-camera transfer added useful
localization on all 60 authored recolor cases and 11 highlights on 60
whole-image adjustments. There were zero flags across 60 unchanged or
recompressed camera controls, including two framing abstentions. These results
support showing the color evidence beside the existing brightness evidence.
The strength of the added highlights depends on whether the earlier file and
any color adjustments are known.
[Development receipt](evaluation-panels/image-reference-webp-color-development-2026-11-15.json.gz) ·
[Held receipt](evaluation-panels/image-reference-webp-color-held-2026-11-15.json.gz) ·
[Camera receipt](evaluation-panels/image-reference-webp-color-cameras-2026-11-15.json.gz)

## See or replay the check

Set `VERITAS_IMAGE_REFERENCE_LOSSY_WEBP_COLOR=true` alongside the local image
reference and lossy WebP settings. After an image analysis, open **Compare with
an earlier copy** and upload the earlier lossless image. Expand **Highlight
difference pixels** for the overlay. Download the separate comparison PDF or
JSON for the same evidence. The verifier recreates the geometry, mask, and crop
pixels from both source files.

Seven local checkouts used simulated USD 9.99 payments, with zero jobs before
payment. The comparison API, PDF, JSON and source replay agreed on all four
framing profiles. One file produced separate color and brightness regions.
A whole-image tint produced zero regions and an unrelated photo abstained.
[Local application receipt](evaluation-panels/image-reference-webp-color-app-2026-11-15.json)

The standard-library reader recomputes the published counts with no weights
or downloads. For native replay, hydrate the pinned BG-REAL sources using
the linked photo evaluation instructions and prepare the earlier CC0
`medium-ordinary` JPEG files by following the linked camera study.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
module=scripts.image_reference_webp_color_transfer
held=docs/evaluation-panels/image-reference-webp-color-held-2026-11-15.json.gz
python3 -B -S -m "$module" review "$held"
backend/.venv/bin/python -B -m "$module" held "$run/held.json.gz" --sources "$photos" --cameras "$cameras"
backend/.venv/bin/python -B -m scripts.image_reference_webp_color_app "$held" "$run/app" --sources "$photos" --cameras "$cameras"
```
