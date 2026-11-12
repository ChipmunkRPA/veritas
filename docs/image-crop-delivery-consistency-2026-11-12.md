# Regional image scores after controlled JPEG recompression

Requiring a PNG crop response to persist after JPEG75 conversion found **6 of 62** known local edits, down from **16 of 62** with the existing remote-crop rule. As a fallback it added **zero** useful locations and **24** ordinary highlights. The candidate stays experimental. [Audit and results](evaluation-panels/image-crop-delivery-consistency-2026-11-12.json)

## Frozen comparison

For one submitted PNG, use the existing 25 native crop positions. Compare the original response with a second response after converting the same full image to JPEG quality 75, before extracting the crops. Center the 25 responses within each delivery at their own median. Each crop receives the smaller of its two centered responses, and the highest crop is eligible. The strict point is the second-highest of eight separate source maxima, each from an untouched photo and its matched whole-image processing control. The fallback retains the native remote-crop finding when one exists and uses the paired result otherwise. The check uses the same weights twice and measures stability under a change of encoding. [Pinned MIT checkpoint and prior native rule](image-remote-crop-editor-transfer-2026-11-11.md)

The comparison used three groups from the previous transfer. All groups were exposed to the project before this test, and the third group contains the seven most recently acquired source photos. All held groups are separate from the Stable Diffusion 1.5 point-fitting photos by recorded source ID and photographer account. The held groups use SDXL and Kandinsky. Each held source supplies a sharp edit, a feathered edit and six ordinary controls. Both deliveries use the exact precompression changed-pixel target. A useful location requires at least 0.30 intersection over union and 0.50 target recall. [Source qualification and rights](image-residual-patch-transfer-2026-11-11.md)

## Results

| Group | Check | Sharp | Feathered | Ordinary highlights |
| --- | --- | ---: | ---: | ---: |
| Earlier 12 sources | Native remote | 3/12 | 1/12 | 34/72 |
| | Paired minimum | 1/12 | 0/12 | 25/72 |
| | Native with fallback | 3/12 | 1/12 | 41/72 |
| Later 12 sources | Native remote | 4/12 | 2/12 | 16/72 |
| | Paired minimum | 1/12 | 0/12 | 20/72 |
| | Native with fallback | 4/12 | 2/12 | 26/72 |
| Recently acquired 7 sources | Native remote | 4/7 | 2/7 | 7/42 |
| | Paired minimum | 2/7 | 2/7 | 7/42 |
| | Native with fallback | 4/7 | 2/7 | 14/42 |

The paired rule removed 29 ordinary flags and added 24 different ones. The fallback's new flags included three untouched photos. It needed a new useful location in each panel and no more than two added ordinary flags per larger panel or one in the seven-source panel. It failed on all three. The test assesses a PNG first and uses JPEG as an intentional second view. It does not establish localization performance on an uploaded JPEG with unknown compression history.

The replay rebuilt 32 fitting and 496 held encoded inputs and checked all 13,200 prepared crop hashes before using the exact stored responses. A second complete replay reproduced the frozen point and all outcomes. A separate scalar tally checked the useful-location and added/lost counts on all 248 held PNG/JPEG pairs. There were no new neural forwards. The source media and per-image responses stay local.

## Reproduce

Run from the repository root with the retained crop-overlap and remote-crop runs, four source intakes and a fresh local output directory. The `development` folder is the eight-source Stable Diffusion 1.5 intake. `fresh` is the seven-source intake added in the residual-patch study.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for phase in freeze fit confirm verify; do
  backend/.venv/bin/python -B -m scripts.image_crop_delivery_consistency "$phase" \
    "$prior" "$remote" "$development" "$earlier" "$later" "$fresh" "$output" || exit 1
done
```
