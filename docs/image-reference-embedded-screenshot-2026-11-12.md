# Earlier-copy comparison for a photo inside a screenshot

The new comparison located 6 of 12 known local edits in real Chromium screenshots and 1 of 6 edits in separate screenshot fixtures. The current earlier-copy comparison could align none of these pairs. The candidate missed both predeclared coverage goals and stays outside the app. It requires an earlier copy and assigns zero AI-origin weight. [Result and native audit](evaluation-panels/image-reference-embedded-screenshot-2026-11-12.json)

A subsequent boundary-witness method passed a separate 18-source panel and is available through the local earlier-copy tool. [Follow-up method and results](image-reference-screenshot-holdout-2026-11-12.md)

The comparison first uses exact RGB anchors to find potential integer offsets between the earlier photo and a PNG screenshot. At each offset it isolates a rectangular inset using abrupt changes in pixel agreement at the four edges. An accepted placement needs exact agreement along at least 80% of each inner edge, independently corroborated textured tiles across the inset, and a compact difference region. The original 90% whole-inset agreement setting screened out known edits whose unchanged areas occupied only 78–87% of the inset. The confirmation arm permits 75% exact agreement inside the rectangle, keeping the edge, tile and uniqueness requirements. Every reported region has both submitted and earlier-copy coordinates. Screenshot content outside the photo is excluded.

The development panel contained 12 actual Chromium groups with LaMa edits and six SDXL/Kandinsky groups with authored screenshot fixtures. The strict method found 7/12 and 0/6 edits respectively, with zero incorrect placements. On those exposed cases, the lower interior agreement setting found four of the six SDXL/Kandinsky edits. The separate confirmation retained this change and compared both settings. The confirmation sources share no exact original RGB, credited Flickr identity or recorded contributor with the development panel. These source photos have appeared in earlier project studies. The screenshot deliveries and their coordinate tests are new. The background-restored edits and unchanged pixels come from the previously qualified source materials. [LaMa intake](evaluation-panels/image-bg-real-backup-intake-v262-complete.json) · [SDXL and Kandinsky intake](image-wavelet-region-transfer-2026-11-11.md)

| Confirmation outcome | Current comparison | Strict inset proof | Revised inset proof |
| --- | ---: | ---: | ---: |
| Actual Chromium edits located /12 | 0 | 6 | 6 |
| SDXL/Kandinsky fixture edits located /6 | 0 | 0 | 1 |
| Clean Chromium photo placements /12 | 0 | 9 | 9 |
| Clean fixture photo placements /6 | 0 | 3 | 3 |
| Accepted incompatible pairs /48 | 0 | 0 | 0 |

Useful localization requires exact alignment, at least 0.30 intersection-over-union, and at least 0.50 recall against native changed pixels. No clean view was highlighted. The revised method also located one ordinary local tone adjustment. Incompatible cases include a duplicate photo placement, an unrelated inset, missing photo content, resized and JPEG versions, and global tone changes. The 48 cases reuse photos across these controls. The confirmation needed at least seven useful Chromium edit locations and four useful alternate edit locations. Median candidate runtime was 33 ms on the Mac, with a 56 ms maximum. Source photos, screenshot files and private case records stay local.

## Replay

Set `alternate` to the retained qualified SDXL/Kandinsky holdout and `run` to a new private directory. The first development run and its original source snapshots remain in the local study archive. Use the existing primary and backup source paths or supply the equivalent retained directories as command flags.

```sh
set -e
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
study() { backend/.venv/bin/python -B -m scripts.image_reference_embedded_screenshot_transfer "$@"; }
study prepare-browser "$run" --confirmation --browser "$run/browser"
node backend/scripts/capture_image_crop_multiscale_transfer.cjs "$run/browser"
for stage in freeze run; do
  study "$stage" "$run" --confirmation --browser "$run/browser" --alternate "$alternate"
done
backend/.venv/bin/python -B -m scripts.review_image_reference_embedded_screenshot_transfer run \
  "$run" --confirmation --browser "$run/browser" --alternate "$alternate"
```
