# Localize differences between a JPEG and its embedded preview

On the held panel, Chipmunk Verify located **16 of 18 known local edits** and **12 of 18 ordinary color retouches**. It highlighted none of the 72 routine controls or 15 views from five genuine published photographs. It declined all 54 incompatible or missing-preview cases. The app can show paired crops and coordinates for an upright JPEG that retains a usable thumbnail. It gives those differences zero AI authorship weight. [Audited results and credited originals](evaluation-panels/image-thumbnail-localization-2026-11-12.json)

The JPEG’s own thumbnail provides a second view of the same scene. The method aligns it to two independently resized views of the main image and estimates ordinary global color changes before measuring localized brightness and color residuals. Only regions corroborated at both resize settings remain. The main crop uses native JPEG coordinates projected from the thumbnail grid. The thumbnail crop covers the original pixels used by both subpixel registrations. Crop previews over 384 pixels on the longest side are reduced for display, with their actual display dimensions recorded separately from the native bounds. The original JPEG and thumbnail may have been edited or saved at different points. The comparison measures their differences at thumbnail resolution. [Research on thumbnail formation](https://erickee.com/papers/spie10.pdf)

| Held outcome | Useful locations or acceptance |
| --- | ---: |
| Feathered LaMa edits | 10/12 |
| Feathered SDXL edits | 3/3 |
| Feathered Kandinsky edits | 3/3 |
| Ordinary near-isoluminant color retouches | 12/18 |
| Routine controls with any highlight | 0/72 |
| Genuine-photo views with any highlight | 0/15 |
| Incompatible or missing-preview cases accepted | 0/54 |

Useful localization requires at least 30% intersection-over-union and 50% recall against the exact pre-encode changed pixels. The luma-only arm located 16 generated edits and zero color retouches. The existing global grayscale comparison marked one generated image and four routine variants from the same source as anomalies, and it supplied a visual difference strip. Its existing behavior remains available. Routine controls include the unedited main image, JPEG75, global tone adjustment, and a resize round trip. The five genuine published photographs supply their original JPEGs with embedded camera thumbnails, plus a recompression and an ordinary white-balance variant. The five sources carry commercial reuse licenses. The 15 views reuse those sources. [LaMa intake](evaluation-panels/image-bg-real-backup-intake-v262-complete.json) · [SDXL and Kandinsky intake](image-wavelet-region-transfer-2026-11-11.md)

The first pilot found 8/12 generated edits and 11/12 color retouches, with 7/48 routine flags on two groups. Those routine differences occupied at most 0.24% of the preview. The separate confirmation froze a 0.4% component area floor. Its 18 edit sources share no recorded contributor, Flickr photo ID, or exact original RGB with the pilot sources. Some sources had prior project exposure in other checks. The generated variants are controlled JPEG fixtures that embed an unchanged preview from the paired earlier source. The five genuine JPEGs were retained across both stages. This panel measures localization under those conditions and does not estimate how often customer JPEGs retain an older, usable thumbnail.

The native review verified 177 case commitments, 354 mask and area checks, and 55 mapped regions. Moving the detector into the runtime package changed only the study runner’s import. The new freeze reproduced every input, truth mask, measured region, output mask, and outcome. The optional evidence runs after image scoring and uses the existing paired exhibit assets. Standard and Summary PDFs and the JSON export retain the recorded geometry. The customer’s existing metadata and confidence calculations remain unchanged.

Three local Full analyses used simulated checkout. The edited JPEG showed a 46 × 65 pixel main-image crop at (73, 406) and an 11 × 15 pixel embedded-preview crop at (15, 88). Both crops matched their original pixels in the public export assets, Standard PDF, and Summary PDF. Chrome displayed the two crops in the plain view. The matched unedited JPEG and the delivery with no preview produced no local preview-difference record.

## Replay

Use the retained qualified image source archives and the genuine JPEGs listed in the receipt. Set `alternate` to the qualified SDXL/Kandinsky holdout and `run` to a new private directory. The initial freeze and sources remain in the local evaluation archive. Source photographs and derived media stay local.

```sh
set -e
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for stage in freeze run review; do
  backend/.venv/bin/python -B -m scripts.image_thumbnail_localization_transfer "$stage" "$run" \
    --confirmation --alternate "$alternate"
done
```
