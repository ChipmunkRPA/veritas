# Native PNG screening and screenshot false alarms

The whole-image check flagged all 16 fully generated PNGs and none of 12 photo controls at its existing operating point. It also flagged six of eight ordinary browser captures, spanning three of four layouts. The frozen false-alarm rule failed and the app keeps this check scoped to JPEG. [Exact result](evaluation-panels/image-whole-png-transfer-2026-11-12.json)

| Held PNG source | Flagged / tested |
| --- | ---: |
| FLUX.1-schnell | 5/5 |
| SDXL | 5/5 |
| Gemini | 6/6 |
| PNGs developed directly from RAW photo captures | 0/4 |
| Lossless PNG exports of ordinary JPEG photos | 0/8 |
| Native Chrome captures of ordinary pages | 6/8 |

The generated files retain the publishers’ original PNG bytes. The T4 benchmark contributes five different prompts each from FLUX.1-schnell and SDXL at 1,024 × 1,024. Its card declares Apache 2.0 for the former outputs and CreativeML Open RAIL++-M for the latter. The CompoSET publication supplies six disjoint fully generated base scenes under CC BY 4.0. Its paper identifies the generator as Gemini. The retained CompoSET deliveries measure 1,408 × 768. That size was matched in the photo controls before scoring. This evaluation uses published outputs only. [Pinned T4 card](https://huggingface.co/datasets/MrAlexGov/t4-imagegen-llm-judges/blob/9136726108ac95b1aabdf200c0969f6b3b39c1d1/README.md) · [Pinned CompoSET card](https://huggingface.co/datasets/CompoSET/CompoSET/blob/d428bf4f23f50ababb205185f179737cd9638db2/README.md) · [CompoSET generation method](https://arxiv.org/html/2610.41118)

The camera stratum uses four exact RAW sensor captures. The pinned decoder reproduces the earlier source pixels, then copies a native rectangle without resampling or JPEG compression before encoding it as PNG. Another eight attributed ordinary photos from a separate source roster are decoded from JPEG and exported losslessly as PNG, with exact native crop geometry recorded where applicable. Four authored browser layouts supply light and dark native captures. Exact hashes and near-image checks exclude the previous whole-image panels. Browser variants share layout groups. Unknown upstream model exposure remains possible. [RawNIND dataset and terms](https://doi.org/10.14428/DVN/DEQCIM) · [Credited JPEG source archive](image-affine-native-scale-2026-11-11.md) · [Earlier JPEG comparison](image-whole-fresh-holdout-2026-11-12.md)

The inherited point was sealed before scoring. Admission required at least 12 generated flags, at least three from each generator, zero RAW-photo flags, and at most one flagged control group. The three flagged browser groups were a reading page, a dashboard, and a form. The experiment stopped at that rule. A paired Full baseline and an app PNG rollout would require passing it first. The source comparison measures one candidate’s flags and provides no calibrated accuracy claim. Its image views show the pixels inspected by the whole-image check.

The CPU process completed all 36 cases in 87.6 seconds with 3.82 GB peak sampled RSS. The independent native readback rebuilt all 108 exact input crops and source coordinates. The source freeze is `ad3fbcde18e10efa7dbacc41ba3b681af8fd68dfb59dc46d0c541169c2373e2c`. Photos and generated pixels stay local.

## Existing coverage rules on photo-rich layouts

All eight simple browser captures in the first panel exceed the app’s existing flat-block limit. A separate CPU probe challenged that result with 12 new browser captures and 12 matched photo PNGs. Six layout groups used 1,366 × 768 and 1,408 × 768 viewports. Four layouts displayed a large ordinary photo with controls, text or map graphics. A dense chart and a flat calendar supplied two graphics-only controls. Six differently credited photos supply the exact source-matched controls. The existing rules needed to exclude at least five screenshot groups at both sizes while excluding at most one photo group. [Exact scope probe](evaluation-panels/image-whole-png-scope-probe-2026-11-12.json)

| Coverage rule | Screens excluded /12 | Photo views excluded /12 | Screenshot groups excluded at both sizes /6 |
| --- | ---: | ---: | ---: |
| Existing flat-block limit or its minimum floor | 2 | 0 | 1 |
| Existing screen-size rule | 6 | 6 | 0 |
| Flat-block rule or screen-size rule | 7 | 6 | 1 |

The flat-block rule excluded the calendar in both sizes. The screen-size rule excluded 1,366 × 768 views in both groups, including one view of every ordinary photo. No rule passed. The native repeat reproduced all 24 PNG hashes, exact flat-block shares, and 120 rule decisions. The probe used the app’s existing content measurements and ran no model inference. PNG remains outside this supplementary check’s serving scope.

## Replay

```sh
python3 -I -S backend/scripts/review_image_whole_png_transfer.py \
  docs/evaluation-panels/image-whole-png-transfer-2026-11-12.json

export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.image_whole_png_transfer acquire "$run"
node backend/scripts/capture_image_whole_png_controls.cjs "$run"
backend/.venv/bin/python -B -m scripts.image_whole_png_transfer freeze "$run"
tmp/image-smaug-v7-env/bin/python -B -m scripts.image_whole_png_transfer candidate "$run"
backend/.venv/bin/python -B backend/scripts/review_image_whole_png_transfer.py \
  "$run/result.json" --native "$run"
```

```sh
python3 -I -S backend/scripts/image_whole_png_scope_controls.py review \
  docs/evaluation-panels/image-whole-png-scope-probe-2026-11-12.json

backend/.venv/bin/python -B -m scripts.image_whole_png_scope_controls prepare "$scope_run"
node backend/scripts/capture_image_whole_png_scope_controls.cjs "$scope_run"
backend/.venv/bin/python -B -m scripts.image_whole_png_scope_controls freeze "$scope_run"
backend/.venv/bin/python -B -m scripts.image_whole_png_scope_controls measure "$scope_run"
```
