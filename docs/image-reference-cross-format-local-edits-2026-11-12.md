# Earlier JPEG evidence for cropped PNG edits

A paired comparison can measure where two supplied images differ. The existing application supports several exact PNG and JPEG resize deliveries. This test examines a cropped and resized RGB PNG made from pixels with a different JPEG history than the available earlier copy. Ordinary compression can change pixels in this setting. The difference between the files carries zero weight in AI authorship confidence. [Current earlier-copy comparison](image-reference-crop-resize-transfer-2026-11-11.md)

Recent research shows that restoring the original pixels around an AI edit can remove whole-image cues that detectors rely on. Separate regional evidence tied to an earlier copy addresses a different question by locating measured changes. The image localizer reviews already tested several single-upload approaches against feathered inserts and ordinary processing. [Inpainting Exchange study](https://arxiv.org/html/2602.00192v2) · [Reference-based photographic assessment](https://arxiv.org/abs/2611.10908) · [Existing localizer results](image-independent-segmenter-transfer-2026-11-11.md)

## Frozen cross-format test

The fixed comparison uses twelve photo and creator groups, separated from the prior development set. The retained source qualification includes six SDXL and six Kandinsky publisher-derived edit groups with exact native source and truth commitments. The source photos have prior project exposure. Each earlier copy is an upright 4:2:0 JPEG encoded with the standard quality-95 tables. The submitted PNG contains an asymmetric crop with a proportional resize. The tested roles include sharp and feathered AI-derived patches, smooth ordinary color changes, local blur, ordinary recompression and contrast, and incompatible geometry or another photo. The matched earlier control uses the exact decoded earlier JPEG as its source. The 144 views reuse twelve sources. [Original source qualification and rights](image-background-patch-transfer-2026-11-11.md) · [Separate photo cohort](image-wavelet-region-transfer-2026-11-11.md)

Require an exact native earlier crop for a useful location. The submitted mask must reach at least 0.30 intersection-over-union and recover at least half the known changed pixels. The gate requires at least six added feathered regions, four added smooth color changes, eight exact matched-earlier negatives, zero wrong crops, and at most four nuisance overlays across three source groups. The runner freezes both files and the submitted-grid truth before measuring. The crop registration and residual settings stay fixed. [Frozen comparison runner](../backend/scripts/image_reference_png_jpeg_transfer.py)

| Input condition | Views | Current useful or matched | Proposed useful or matched | Added nuisance overlays |
| --- | ---: | ---: | ---: | ---: |
| Sharp AI-derived patch | 12 | 0 | 10 | |
| Feathered AI-derived patch | 12 | 0 | 10 | |
| Smooth ordinary local color | 12 | 0 | 11 | |
| Exact earlier JPEG pixels resized without a local edit | 12 | 8 | 11 | 0 |
| PNG from unchanged pixels with separate JPEG history | 12 | | | 2 |
| Prior JPEG compression | 12 | | | 2 |
| Whole-image contrast adjustment | 12 | | | 1 |
| Rotation, fractional shift, stretch or another photo | 48 | 0 accepted | 0 accepted | 0 |

The broader cross-format candidate missed the nuisance gate by one view. All five nuisance overlays came from three source groups. The largest unchanged-delivery overlay covered 8,177 submitted pixels. Every accepted crop matched the intended four native edges. The comparison also highlighted eight ordinary local blur deliveries, with zero useful blur locations under the fixed overlap criterion. Native readback verified 118 source crops and 59 overlays. These outcomes preserve the candidate as research evidence.

An exploratory test on these already scored files retained only pixels with a local RGB difference of at least 10 levels after robust whole-image tone adjustment under all three resize kernels. It retained all 10 useful sharp, 10 useful feathered and 11 useful smooth color changes and removed all five nuisance overlays. It also removed all eight local blur overlays. The stricter measurement therefore concerns substantial local color differences. An unchanged area on its map can still contain smaller changes or blur.

## Separate color stability check

The follow-up fixes the 10-level comparison before scoring the new inputs. Six exact FLUX edits use distinct photo and contributor groups from the earlier comparison. Six additional Wikimedia photo groups supply ordinary controls. The FLUX groups had earlier project exposure, including a preliminary run with the unfiltered crop and resize comparison. The stricter all-kernel pixel test was set using the earlier SDXL and Kandinsky panel. The 120 pairs contain original, recompression, exposure, known local color and blur, incompatible framing and unrelated-photo controls. [Exact FLUX construction and masks](image-flux-photo-expansion-v285-result.md) · [Wikimedia source credits](image-jpeg-public-replay-2026-11-12.md) · [Stability runner](../backend/scripts/image_reference_png_jpeg_stability.py)

| Follow-up result | Existing app | Cross-format candidate | With the stability requirement |
| --- | ---: | ---: | ---: |
| Useful sharp FLUX regions | 0/6 | 4/6 | 2/6 |
| Useful feathered FLUX regions | 0/6 | 3/6 | 0/6 |
| Useful ordinary smooth color regions on the FLUX source photos | 0/6 | 4/6 | 4/6 |
| Useful ordinary smooth color regions on the public photos | 0/6 | 6/6 | 6/6 |
| Ordinary delivery nuisance overlays | 0/36 | 5/36 | 0/36 |
| Incompatible pairs accepted | 0/48 | 0/48 | 0/48 |

The stability requirement suppressed every ordinary delivery nuisance and localized ordinary color retouching. It missed both prespecified generator criteria, which each required three useful locations. The first printed summary also mislabeled the ordinary smooth-color count. Correcting that label records the six measured public-photo successes and leaves the failed generator gate intact. The native reviewer verified 34 crops and 17 overlays. Both candidates remain experimental. [Aggregate, freeze and exact replay receipts](evaluation-panels/image-reference-cross-format-local-edits-2026-11-12.json)

## Reproduce

From the repository root, set `development` and `held` to the retained [first intake](image-background-patch-transfer-2026-11-11.md) and [separate intake](image-wavelet-region-transfer-2026-11-11.md) folders. Use new scratch paths for both runs. The second runner uses the existing FLUX edit archive and the exact licensed JPEG deliveries from the [public replay acquisition](image-jpeg-public-replay-2026-11-12.md). Set `public_photos` to the public replay folder.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for stage in freeze run review; do
  backend/.venv/bin/python -B -m scripts.image_reference_png_jpeg_transfer \
    "$stage" "$development" "$held" "$cross_format_run" || exit
  backend/.venv/bin/python -B -m scripts.image_reference_png_jpeg_stability \
    "$stage" "$public_photos" "$held/plan.json" "$stability_run" || exit
done
```
