# Local differences in PNGs that retain an Exif preview

Chipmunk Verify can compare the main image of a supported PNG with the smaller JPEG preview saved inside it. When the views align and show a localized difference, the results page and PDF show the main crop, stored preview crop and measured footprint. The JSON export and signed evidence package preserve the coordinates and image commitments. The preview is part of the submitted file. Its date and origin are unknown. Ordinary retouching can produce the same evidence. The comparison has zero weight in the AI authorship assessment. [PNG eXIf specification](https://www.w3.org/TR/png-3/#eXIf)

The separate opt-in is `VERITAS_IMAGE_PNG_EXIF_PREVIEW_REVIEW=true`. It supports upright static 8-bit RGB PNGs up to 24 megapixels, with valid CRCs and a single eXIf chunk before image data and an upright RGB JPEG preview with sides between 64 and 512 pixels. Untagged images and standard sRGB declarations are supported. Alpha, animation, palette images, custom profiles, conflicting or nonstandard color declarations, malformed metadata and incompatible previews are declined. Images with non-upright effective orientation are also declined, including orientation read from XMP. The comparison runs only where the existing signed-provenance stop allows it.

## Controlled local results

The test used 18 independently attributed originals and exact native edit masks from two retained source panels. Each PNG carried a preview rendered from the corresponding earlier pixels. Routine deliveries covered untouched images, global brightness and resize round trips. Matched local tone and blur use the same native footprint as the known generated edit. The source photos and generated edits had earlier project evaluation exposure. The PNGs were constructed for this test, so the study provides no estimate of how often an actual upload retains such a preview. [BG-REAL source rights](image-bg-real-roster-v218-method.md) · [Retained FLUX compositions](image-flux-photo-expansion-v285-result.md)

| Exact outcome | Result |
| --- | ---: |
| Sharp BG-REAL edits localized | 11/12 |
| Feathered BG-REAL edits localized | 11/12 |
| Sharp and feathered small FLUX edits localized | 0/12 |
| Known ordinary local color changes localized | 4/18 |
| Known ordinary local blurs localized | 0/18 |
| Routine delivery views highlighted | 0/54 |
| Missing or unrelated previews compared | 0/36 |
| Unsupported format or metadata cases compared | 0/15 |
| Standard sRGB and gamma carrier variants compared | 2/2 |

A useful measured footprint reached at least 30% intersection over union and 50% recall against the exact changed pixels. One FLUX source produced an accurate partial highlight on its sharp and feathered views. Its feathered highlight covered 1,624 of 3,898 changed pixels and no pixels outside that region. The fixed cross-family criterion of three useful feathered FLUX regions failed. The scoped opt-in serves actual differences from the saved preview and supports no general claim of edit or AI detection. [Evaluation receipt](evaluation-panels/image-png-exif-preview-2026-11-12.json)

The isolated Full worker checked an edited PNG, its matched ordinary file, and the same edit with the option disabled. Each simulated Stripe order queued zero analysis jobs before payment and one after payment, including a duplicate paid event. The edited result published one region and three PNG assets. The exhibit API, JSON coordinates, Standard PDF and signed package matched the original PNG, the embedded JPEG and the recorded footprint. The ordinary and opt-out files produced no preview evidence. Chrome loaded all three assets in the plain result and showed no preview card for the ordinary control. [Application receipt](evaluation-panels/image-png-exif-preview-2026-11-12.json)

## Replay

Run this standard-library check from the repository root to recalculate the
179 case decisions and compare all 60 crop commitments with the published
result. It uses the original case ledger, which contains no media or pixel
arrays. Use the native replay below to re-decode the actual crops.

```sh
python3 -B -IS backend/scripts/review_image_png_exif_preview_transfer.py \
  docs/evaluation-panels/image-png-exif-preview-2026-11-12.json \
  docs/evaluation-panels/image-png-exif-preview-cases-2026-11-12.json
```

Use the retained source paths from the runner and a fresh local directory for each command sequence. The native replay reconstructs every input, comparison, exact changed-pixel mask and crop. It leaves original and derived media in the local archive.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for stage in freeze run review; do
  backend/.venv/bin/python -B -m scripts.image_png_exif_preview_transfer "$stage" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.image_png_exif_preview_app "$isolated_app"
backend/.venv/bin/python -m app.cli verify "$zip" --public-key "$public_key" --source-image "$original"
```
