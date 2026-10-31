# Ablation: img.dfdc_b7

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.hf.ateeqq,img.hf.dima806,img.dfdc_b7,img.forensic.spectrum,img.forensic.ela,img.forensic.noise,img.forensic.jpeg_dq,img.forensic.copymove,img.meta.generator,img.meta.exif,img.meta.consistency,img.meta.qtables,img.watermark.dwtdct GATES='img.dfdc_b7=has_face' CONTEXT=scored ABLATE=img.dfdc_b7 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.forensic.copymove, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, img.forensic.spectrum, img.hf.ateeqq, img.hf.dima806, img.meta.consistency, img.meta.exif, img.meta.generator, img.meta.qtables, img.watermark.dwtdct
- Items (fit ∪ cal): 5221 (2694 real, 2527 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05
- Criteria recomputed with D-44 (`app.bench.ablation.criteria`, H-4) from the stored LGO and cross-fit tables; the JSON keeps the D-41 criteria of the run.

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +3.1 pts, CI [+1.8, +4.6] |
| A3b | pass | family faceswap-stills, threat faceswap |
| A4 | pass | pooled reals: Δ FPR +0.0 pts, CI [-0.4, +0.4] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 163 | 58.3 % | 62.6 % | +4.3 |
| F2 | 80 | 58.8 % | 63.7 % | +5.0 |
| F3 | 62 | 38.7 % | 41.9 % | +3.2 |
| F4 | 130 | 59.2 % | 61.5 % | +2.3 |
| F5 | 94 | 34.0 % | 36.2 % | +2.1 |
| F6 | 113 | 74.3 % | 73.5 % | -0.9 |
| F7 | 118 | 76.3 % | 82.2 % | +5.9 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 57.1 %, with 60.2 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 5.7 % | 6.4 % | +0.7 | [+0.0, +2.5] |
| R2 | 611 | 0.2 % | 0.2 % | +0.0 | [+0.0, +0.0] |
| R3 | 587 | 12.3 % | 13.5 % | +1.2 | [+0.3, +2.2] |
| R4 | 495 | 0.6 % | 0.6 % | +0.0 | [+0.0, +0.0] |
| R5 | 549 | 10.9 % | 9.1 % | -1.8 | [-3.3, -0.2] |
| R6 | 28 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 175 | 45.7 % | 49.1 % | +3.4 | [+1.1, +6.3] |
| threat faceswap | 301 | 41.5 % | 67.8 % | +26.2 | [+18.5, +34.3] |
| threat gan_face | 163 | 14.7 % | 19.0 % | +4.3 | [+0.0, +8.6] |
| threat personalised | 124 | 27.4 % | 30.6 % | +3.2 | [+0.8, +6.5] |
| threat t2i | 1764 | 62.3 % | 63.9 % | +1.6 | [+0.9, +2.4] |
| family faceswap-stills | 301 | 41.5 % | 67.8 % | +26.2 | [+18.5, +34.3] |
| family flux1 | 93 | 75.3 % | 75.3 % | +0.0 | [+0.0, +0.0] |
| family flux2 | 80 | 67.5 % | 70.0 % | +2.5 | [-2.5, +7.7] |
| family gan-faces | 163 | 14.7 % | 19.0 % | +4.3 | [+0.0, +8.6] |
| family gpt-4o-image-1 | 46 | 67.4 % | 69.6 % | +2.2 | [+0.0, +6.5] |
| family gpt-image-1.5-2 | 62 | 41.9 % | 43.5 % | +1.6 | [-3.2, +6.5] |
| family hidream | 52 | 94.2 % | 96.2 % | +1.9 | [+0.0, +6.0] |
| family hunyuanimage | 29 | 82.8 % | 82.8 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 52 | 73.1 % | 76.9 % | +3.8 | [+0.0, +9.8] |
| family imagen3 | 60 | 41.7 % | 46.7 % | +5.0 | [+0.0, +11.7] |
| family imagen4 | 118 | 83.9 % | 84.7 % | +0.8 | [-1.8, +3.6] |
| family legacy | 163 | 37.4 % | 36.2 % | -1.2 | [-3.1, +0.0] |
| family long-tail | 203 | 61.1 % | 64.5 % | +3.4 | [+1.1, +6.2] |
| family midjourney | 163 | 62.0 % | 65.6 % | +3.7 | [+1.2, +6.9] |
| family modern-t2i-faces | 124 | 83.9 % | 86.3 % | +2.4 | [+0.0, +5.1] |
| family nano-banana | 130 | 62.3 % | 63.1 % | +0.8 | [-2.3, +3.9] |
| family partial-edits | 175 | 45.7 % | 49.1 % | +3.4 | [+1.1, +6.3] |
| family personalised | 124 | 27.4 % | 30.6 % | +3.2 | [+0.8, +6.5] |
| family qwen-image | 113 | 74.3 % | 75.2 % | +0.9 | [-1.8, +4.4] |
| family recraft | 57 | 61.4 % | 61.4 % | +0.0 | [+0.0, +0.0] |
| family sd3 | 84 | 46.4 % | 46.4 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 41 | 51.2 % | 51.2 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 94 | 35.1 % | 37.2 % | +2.1 | [+0.0, +5.4] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
