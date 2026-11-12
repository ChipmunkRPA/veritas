# Test smaller image regions on known edits and ordinary layouts

The prior one-third crop could rarely fit a feathered photo edit or an edit embedded in a browser screenshot. This study freezes smaller source rectangles and tests whether the same image classifier can select useful locations. The comparison uses the earlier experimental crop rule as its reference. [Earlier crop and layout study](image-crop-layout-transfer-2026-11-12.md)

## Frozen method

Generate rectangles spanning one-sixth, one-quarter and one-third of the native source width and height. A half-window stride covers each edge, yielding 121, 49 and 25 proposed regions. Positions use exact integer bounds. A crop carries its source coordinates through measurement. The model receives a bilinear 384 × 384 RGB view of each crop. The three proposal sets depend only on source dimensions. No edit mask or text recognition enters the model or ranking.

At each scale, subtract the median crop response from that scale's maximum. The largest of the three contrasts determines the single proposed region. Use one strict internal point at the second-highest of eight ordinary fitting group maxima. The earlier comparison uses its original one-third rule and median of nonoverlapping crops, with both historical points fixed. The original checkpoint is MIT licensed. This research score has zero weight in authorship confidence. [Pinned checkpoint](https://huggingface.co/OwensLab/commfor-model-384/tree/6076002bf0d9dd37537f965ee2f06f826c333b61) · [Earlier crop reference](image-remote-crop-editor-transfer-2026-11-11.md)

Recent inpainting research finds that a whole-image detector can respond to generation-related changes outside the edited region. Pixel-grounded benchmarks also distinguish actual changed pixels from a semantic selection mask. This comparison measures exact changed pixels and gives the detector no mask. No new paper code, checkpoint or dataset is included. [November 2026 inpainting study](https://arxiv.org/html/2602.00192v2) · [Pixel-grounded evaluation](https://arxiv.org/abs/2603.20193)

A useful location requires native changed-pixel intersection over union of at least 0.30 and changed-pixel recall of at least 0.50. The frozen release check requires at least three additional useful feathered photo locations from separate held sources, at least one additional useful embedded edit, every useful comparator location preserved, at most two newly flagged ordinary photo groups, and at most one newly flagged ordinary screenshot group. Group flags count any flagged ordinary view in a group. The full 24 fitting and 136 held views, the criteria, source and truth hashes, code, dependencies and weight digest were frozen before scoring.

## Photo and layout sources

The eight fitting contributors are distinct within the credited Commons archive and separate from the eight sources used in the preceding crop and layout study. Each fitting group has an original, a global tone adjustment and ordinary blur. The held sources include 12 exact LaMa object-removal pairs, with three from each of four publisher splits. They come from the retained set of 24 primary source groups. The previous crop and layout study used a separate set of 12 backup groups. Each primary photo is evaluated in its original, edited, feathered, tone-adjusted, locally blurred and resampled form. The source photos are licensed CC BY 2.0, the publisher's contributions use CC BY 4.0, and the publisher records Apache-2.0 LaMa. [Commons source terms](image-affine-native-scale-2026-11-11.md) · [Primary pairs and exact masks](image-bg-real-transport-v225-result.md) · [Held source roster terms](image-bg-real-roster-v218-method.md)

Seven previously exposed SDXL and Kandinsky source groups test preservation of known wider crop locations. They are reported separately from the 12 primary groups. The four new authored Chromium layouts each contain three groups with an ordinary interface, an ordinary interface with a genuine photo, and the exact same interface with the feathered photo edit. The photo is rendered at its original scale. Every changed screenshot pixel lies inside the shown photo and matches the paired source difference. The screenshots use the same 12 primary source groups, so the photo and screenshot results are correlated. The photo archive and all rendered media remain local. The primary image sources had earlier project exposure in different experiments. The separation is from this fit and the prior crop and layout trial. [Legacy edit pairs](image-residual-patch-transfer-2026-11-11.md)

## Dimension-only limits

| Positive input | Useful proposal at one-third | One-quarter | One-sixth | Any scale |
| --- | ---: | ---: | ---: | ---: |
| Primary edited photos | 3/12 | 6/12 | 8/12 | 10/12 |
| Primary feathered photos | 3/12 | 5/12 | 8/12 | 10/12 |
| Feathered edits inside interfaces | 2/12 | 4/12 | 6/12 | 8/12 |
| Legacy edited photos | 7/7 | 7/7 | 0/7 | 7/7 |
| Legacy feathered photos | 7/7 | 7/7 | 0/7 | 7/7 |

These are geometry ceilings from exact masks and all dimension-only proposals. They measure what the crop search can represent before model ranking or flagging.

## Held model results

The combined rule failed its frozen criteria and remains experimental. It found no useful locations on the 12 held primary feathered photos or the 12 embedded edits. It missed the same primary source in both edited forms where the earlier research rule had useful locations. On the seven historical sources it gained one sharp and one feathered location from the same source group. [Audited scalar ledger](evaluation-panels/image-crop-multiscale-transfer-2026-11-12.json)

| Known edits | One-third research rule, flags | Useful | Multi-scale, flags | Useful |
| --- | ---: | ---: | ---: | ---: |
| Primary edited photos | 3/12 | 1/12 | 2/12 | 0/12 |
| Primary feathered photos | 3/12 | 1/12 | 1/12 | 0/12 |
| Feathered edits inside interfaces | 3/12 | 0/12 | 1/12 | 0/12 |
| Legacy edited photos | 5/7 | 4/7 | 5/7 | 5/7 |
| Legacy feathered photos | 2/7 | 2/7 | 4/7 | 3/7 |

| Ordinary controls | One-third research rule, flags | Multi-scale, flags |
| --- | ---: | ---: |
| Primary original | 1/12 | 0/12 |
| Primary global tone | 3/12 | 0/12 |
| Primary local blur | 1/12 | 1/12 |
| Primary resize round trip | 2/12 | 0/12 |
| Legacy original | 0/7 | 0/7 |
| Legacy global processing | 0/7 | 3/7 |
| Total ordinary photo views | 7/62 | 4/62 |
| Ordinary interface only | 0/12 | 0/12 |
| Ordinary interface with a genuine photo | 2/12 | 0/12 |
| Total ordinary screenshot views | 2/24 | 0/24 |

Both rules flagged four of 19 ordinary photo groups. The four candidate groups were all newly flagged, including three historical global-processing controls. This exceeds the frozen limit of two newly flagged photo groups. The candidate cleared both previously flagged ordinary interface groups and added zero.

The model's ranking used little of the expanded search. Even before the operating point, the combined winner was useful for only 1/12 primary feathered photos and 2/12 embedded edits. Considering each scale's own top crop raises both to 3/12. None of those useful per-scale responses passes the combined point. The raw combined winner had useful coordinates for both views of the lost primary source and scored below the fitted point. More proposal sizes alone therefore address only one part of the localization problem. The next comparison needs a response trained or independently supported on source-preserving regional edits, with ordinary controls and unchanged held groups for readback.

The separate existing sharp-boundary algorithm measured 11 of 12 primary groups and supplied no useful primary or embedded location. It flagged 4/48 ordinary primary views, with 44 eligible, 2/14 ordinary legacy views and 24/24 ordinary screenshots. Its useful historical locations were 5/7 native and 1/7 feathered. This diagnostic uses its shipped operating point on eligible dimensions. It does not enter the frozen promotion test, whose one-third comparator is also research-only. The app continues to use its existing image checks.

The CPU fit measured 4,680 inputs in 228 seconds. The held pass measured 14,442 inputs in 719 seconds and reused 12,078 byte-identical crop responses. Peak held RSS was about 1.05 GiB. The freeze is `70a5ff91628d4d337584e320a8607f276bfa5028696255d93149bd2c5cd83318`. The pre-inference geometry is `95a59951e1efb0455bd104c0cb0a5260d4b9dcf6357c9adc65799be4cb6ae6a7`.

The independent audit rebuilt all 160 native views, 31,200 prepared crop inputs and 9,750 positive-case geometry overlaps. All 28 historical one-third selections and decisions matched the earlier journal, with a maximum logit difference below one millionth across 700 values. The cross-role source audit found zero exact or near-thumbnail matches across 236 pairs. These checks do not establish author-alias independence.

## Reproduce

The public reader uses Python's standard library. It recomputes every crop winner, the control point, the earlier comparator and the aggregate from ordered scores and native overlap counts. The native audit also rebuilds the source views and model-input crop bytes, compares the one-third crops with the historical replay, and measures exact overlaps directly from retained masks. It uses saved model outputs. It verifies whole-artifact commitments without redistributing photographs, screenshots, truth masks or individual input hashes.

```sh
python3 -I -S backend/scripts/review_image_crop_multiscale_transfer.py \
  docs/evaluation-panels/image-crop-multiscale-transfer-2026-11-12.json
```

The retained qualified inputs are the Commons affine-scale archive, primary v225 roster and media, backup v262 roster, seven source-preserving legacy edits, previous crop-layout receipt, earlier remote-crop receipt and the pinned MIT checkpoint. The browser capture runs offline. Paths below name those archives in that order.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
backend/.venv/bin/python -B -m scripts.image_crop_multiscale_transfer prepare \
  "$commons" "$primary" "$backup" "$legacy" "$previous" "$baseline" "$weight" "$run"
node backend/scripts/capture_image_crop_multiscale_transfer.cjs "$run"
for stage in freeze geometry fit held; do
  backend/.venv/bin/python -B -m scripts.image_crop_multiscale_transfer "$stage" \
    "$commons" "$primary" "$backup" "$legacy" "$previous" "$baseline" "$weight" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_image_crop_multiscale_transfer "$run/public.json" --native \
  "$commons" "$primary" "$backup" "$legacy" "$previous" "$baseline" "$weight" "$run"
```
