# Test photo-only image evidence inside screenshots

The earlier sharp boundary check often highlights ordinary interface geometry. This study tests whether a separate source-pixel layout measurement can restrict the check to a photograph visible inside a screenshot. The three comparison paths are the whole image, a proposed photo crop, and a known exact photo crop measured as a diagnostic. The known crop separates a routing error from an error made by the regional check inside the right photograph. [Prior layout-confound tests](image-boundary-multiple-transfer-2026-11-12.md)

## Frozen measurement

Measure neighboring RGB changes across both directions of each native 8 × 8 block. Close and open the active block grid, then require one dominant textured component. Refine each of its four sides using a 12-pixel strip on each side of the proposed edge. The inside strip must show a distinct increase in the share of changing pixels over the gutter. Route only when an exterior gutter contains three aligned rows of recognized words and the bounded photo satisfies the existing sharp check's native size requirements. The existing whole-image result stays in use when the route has insufficient evidence. The OCR output is used for word geometry only. [Existing sharp boundary method](image-boundary-prominence-holdout-2026-11-11.md) · [OCR code license](https://github.com/tesseract-ocr/tesseract/blob/main/LICENSE) · [English language data license](https://github.com/tesseract-ocr/tessdata_fast/blob/main/LICENSE)

The October GUI-Lens revision uses OCR and visible component positions as coarse-to-fine coordinate references. A separate recent text-image forensics study focuses forensic localization on smaller areas. Here the native texture and exterior layout geometry are independent of the edit score. The capture oracle supplies exact coordinates for evaluation only. The experiment uses no new downloaded model or upstream paper code. [GUI-Lens, October revision](https://arxiv.org/html/2608.03270v2) · [Text-image forensic localization](https://arxiv.org/html/2609.02097v1)

The boundary check uses its existing strict internal point in every arm. A useful location requires at least 0.30 changed-pixel intersection over union and 0.50 recall. The prospectively sealed gate requires at least eight correct photo preview routes at 0.95 viewport IoU, at most two ambiguous routes, at least six fewer ordinary screenshot flags, no new ordinary screenshot flags, at most one misrouted standalone photo group, at least two new useful sharp screenshot locations, and at most one lost useful location. The measurements have zero authorship confidence weight.

## Source groups

Twelve distinct commercially qualified edited-photo groups come from two existing BG-REAL archives, with six SDXL and six Kandinsky edits. The later six are outside the preceding tile comparison. The source photos have prior project exposure. The rendered screenshot arrangements and exact photo windows are new. Every group includes a text-only layout, a clean photo preview, matched sharp and feathered edited previews, and a preview with a conventional patterned overlay. The latter is an ordinary ambiguous layout. Native browser captures preserve the same photo pixel values in the matched previews. A separate readback checks all outside pixels and the full changed-pixel masks. [Earlier source and terms](image-background-patch-transfer-2026-11-11.md) · [Later qualified sources](image-wavelet-region-transfer-2026-11-11.md)

The exact native photo crops also run alone, alongside ordinary tone variants. Eight additional credited Commons photos provide 16 standalone and tone controls. The held set contains 60 Chromium screenshots, 48 paired standalone crops and 16 separate ordinary photo views. The cross-role original check compared 96 exact and thumbnail pairs between the qualified edits and extra controls. Media and individual image commitments stay local. [Commons acquisition and attribution](image-affine-native-scale-2026-11-11.md)

## Held results

The route correctly bounded seven of 12 clean photo previews and the same number of sharp and feathered previews. No standalone photo group was routed. Five ambiguous interfaces passed the proposed route. The conditioned boundary check removed four ordinary preview flags and found six fewer useful sharp locations. No arm localized a feathered edit. The candidate fails the frozen gate. [Audited scalar ledger](evaluation-panels/image-viewport-routing-transfer-2026-11-12.json)

| Held screenshot input | Whole flags | Routed flags | Known exact crop flags | Whole useful | Routed useful | Known exact crop useful |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary text layout | 6/12 | 6/12 | — | — | — | — |
| Ordinary photo preview | 10/12 | 6/12 | 1/12 | — | — | — |
| Ordinary patterned overlay | 12/12 | 12/12 | — | — | — | — |
| Sharp known edit | 12/12 | 9/12 | 7/12 | 9/12 | 3/12 | 1/12 |
| Feathered known edit | 9/12 | 6/12 | 1/12 | 0/12 | 0/12 | 0/12 |

All 36 exact screenshot crops reproduce the same boundary responses and coordinates as their corresponding standalone views after accounting for the crop offset. The standalone paired controls flag two of 24 ordinary views. The eight additional photographic control groups flag six of 16 views. Whole and routed results agree on all 64 standalone views.

## Why exact cropping lost useful sharp evidence

A separate post-run geometry audit enumerated every allowed rectangle using the native masks. The whole-screenshot sizes, the smaller crop-relative sizes, and the original sizes restricted to the exact photo each offered at least one useful region for every one of the 12 sharp and 12 feathered screenshots. The detector's actual selection remained weak within the exact crop even where an eligible useful rectangle existed.

All nine useful whole-screenshot sharp boxes lie within the photo coordinates. Eight had a side within three pixels of the viewport edge. Cropping removes a neighboring side sample used in their four-side scores. The remaining box reproduces the same four side measurements in the exact crop and its physical size falls outside that crop's allowed rectangle sizes. On matched clean screenshots all nine boxes fell below the review point when evaluated at the same coordinates. The useful whole-screenshot boxes therefore depended on how the true edit cues and the enclosing photo boundary were measured together.

A future independent comparison can vary the screen frame while holding the same source photo and edited pixels fixed. It can retain the native candidate sizes and check for support from the interior sides separately from any frame-coincident side. The current result supports keeping the existing app path.

The held run finished in 19 seconds. The independent readback verified 124 original views, 186 exact crop and four-side measurements, 26 exterior OCR replays and 36 exact viewport versus standalone pairs. The freeze is `9832d5fdaf188c6fd7c3b606d839acfd8bcff406d46a5c1f9e1a14468727fb50`.

## Reproduce

The portable reader recalculates the three arms and the frozen outcome from their scalar records. The native reader rebuilds source crops and calculates the four-side responses directly from neighboring pixels. The fixture and readback require only retained qualified inputs, local Chromium and the existing English OCR installation.

```sh
python3 -I -S backend/scripts/review_image_viewport_routing_transfer.py \
  docs/evaluation-panels/image-viewport-routing-transfer-2026-11-12.json
```

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.image_viewport_routing_transfer prepare \
  "$commons" "$earlier" "$later" "$tesseract" "$tessdata" "$run"
node backend/scripts/capture_image_viewport_routing.cjs "$run"
for stage in freeze run; do
  backend/.venv/bin/python -B -m scripts.image_viewport_routing_transfer "$stage" \
    "$commons" "$earlier" "$later" "$tesseract" "$tessdata" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_image_viewport_routing_transfer "$run/public.json" --native \
  "$commons" "$earlier" "$later" "$tesseract" "$tessdata" "$run"
```
