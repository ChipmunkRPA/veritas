# Ablation: img.forensic.spectrum

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.hf.ateeqq,img.hf.dima806,img.dfdc_b7,img.forensic.spectrum,img.forensic.ela,img.forensic.noise,img.forensic.jpeg_dq,img.forensic.copymove,img.meta.generator,img.meta.exif,img.meta.consistency,img.meta.qtables,img.watermark.dwtdct GATES='img.dfdc_b7=has_face' CONTEXT=scored ABLATE=img.forensic.spectrum ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dfdc_b7, img.forensic.copymove, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, img.hf.ateeqq, img.hf.dima806, img.meta.consistency, img.meta.exif, img.meta.generator, img.meta.qtables, img.watermark.dwtdct
- Items (fit ∪ cal): 5221 (2694 real, 2527 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05
- Criteria recomputed with D-44 (`app.bench.ablation.criteria`, H-4) from the stored LGO and cross-fit tables; the JSON keeps the D-41 criteria of the run.

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro +0.2 pts, CI [-0.2, +0.7] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR +0.0 pts, CI [-0.1, +0.1] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 163 | 62.6 % | 62.6 % | +0.0 |
| F2 | 80 | 63.7 % | 63.7 % | +0.0 |
| F3 | 62 | 41.9 % | 41.9 % | +0.0 |
| F4 | 130 | 60.8 % | 61.5 % | +0.8 |
| F5 | 94 | 36.2 % | 36.2 % | +0.0 |
| F6 | 113 | 74.3 % | 73.5 % | -0.9 |
| F7 | 118 | 80.5 % | 82.2 % | +1.7 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 60.0 %, with 60.2 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 6.4 % | 6.4 % | +0.0 | [+0.0, +0.0] |
| R2 | 611 | 0.2 % | 0.2 % | +0.0 | [+0.0, +0.0] |
| R3 | 587 | 13.5 % | 13.5 % | +0.0 | [+0.0, +0.0] |
| R4 | 495 | 0.6 % | 0.6 % | +0.0 | [+0.0, +0.0] |
| R5 | 549 | 9.1 % | 9.1 % | +0.0 | [-0.5, +0.5] |
| R6 | 28 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 175 | 48.6 % | 49.1 % | +0.6 | [+0.0, +1.7] |
| threat faceswap | 301 | 67.8 % | 67.8 % | +0.0 | [-0.9, +1.0] |
| threat gan_face | 163 | 19.0 % | 19.0 % | +0.0 | [+0.0, +0.0] |
| threat personalised | 124 | 30.6 % | 30.6 % | +0.0 | [+0.0, +0.0] |
| threat t2i | 1764 | 64.2 % | 63.9 % | -0.2 | [-0.5, +0.1] |
| family faceswap-stills | 301 | 67.8 % | 67.8 % | +0.0 | [-0.9, +1.0] |
| family flux1 | 93 | 75.3 % | 75.3 % | +0.0 | [+0.0, +0.0] |
| family flux2 | 80 | 70.0 % | 70.0 % | +0.0 | [+0.0, +0.0] |
| family gan-faces | 163 | 19.0 % | 19.0 % | +0.0 | [+0.0, +0.0] |
| family gpt-4o-image-1 | 46 | 69.6 % | 69.6 % | +0.0 | [+0.0, +0.0] |
| family gpt-image-1.5-2 | 62 | 43.5 % | 43.5 % | +0.0 | [+0.0, +0.0] |
| family hidream | 52 | 96.2 % | 96.2 % | +0.0 | [+0.0, +0.0] |
| family hunyuanimage | 29 | 82.8 % | 82.8 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 52 | 76.9 % | 76.9 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 60 | 46.7 % | 46.7 % | +0.0 | [+0.0, +0.0] |
| family imagen4 | 118 | 85.6 % | 84.7 % | -0.8 | [-2.7, +0.0] |
| family legacy | 163 | 36.8 % | 36.2 % | -0.6 | [-2.0, +0.0] |
| family long-tail | 203 | 64.5 % | 64.5 % | +0.0 | [+0.0, +0.0] |
| family midjourney | 163 | 65.6 % | 65.6 % | +0.0 | [-1.8, +1.9] |
| family modern-t2i-faces | 124 | 87.9 % | 86.3 % | -1.6 | [-4.1, +0.0] |
| family nano-banana | 130 | 63.1 % | 63.1 % | +0.0 | [+0.0, +0.0] |
| family partial-edits | 175 | 48.6 % | 49.1 % | +0.6 | [+0.0, +1.7] |
| family personalised | 124 | 30.6 % | 30.6 % | +0.0 | [+0.0, +0.0] |
| family qwen-image | 113 | 75.2 % | 75.2 % | +0.0 | [+0.0, +0.0] |
| family recraft | 57 | 61.4 % | 61.4 % | +0.0 | [+0.0, +0.0] |
| family sd3 | 84 | 46.4 % | 46.4 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 41 | 51.2 % | 51.2 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 94 | 37.2 % | 37.2 % | +0.0 | [+0.0, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
