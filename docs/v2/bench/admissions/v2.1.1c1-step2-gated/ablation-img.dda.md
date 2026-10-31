# Ablation: img.dda

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7,img.hf.ateeqq,img.hf.dima806 GATES='img.dda=!has_face;img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.dda ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dfdc_b7, img.hf.ateeqq, img.hf.dima806
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (MSCOCO train2017)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3
- Operating point (D-53): reals scoring exactly τ count as flagged with weight q (the rank rule's share); served rule (s ≥ τ) weighted FPR: without 5.0 % (q 0.165), with 5.0 % (q 0.626)

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +6.7 pts, CI [+4.2, +9.3] |
| A3b | pass | family flux1, family gpt-4o-image-1, family ideogram, family imagen4, family midjourney, family qwen-image, family recraft, family sd3, family seedream3, threat t2i |
| A4 | pass | pooled reals: Δ FPR +0.0 pts, CI [-0.8, +0.8] (blocks with a lower bound above +1.0, D-53); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R5; above +2.0 on fewer than 150 reals (no term): R6 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 69.3 % | 80.4 % | +11.1 |
| F2 | 70 | 24.3 % | 32.9 % | +8.6 |
| F3 | 57 | 10.5 % | 7.0 % | -3.5 |
| F4 | 118 | 28.8 % | 36.4 % | +7.6 |
| F5 | 80 | 63.7 % | 73.8 % | +10.0 |
| F6 | 106 | 34.9 % | 37.7 % | +2.8 |
| F7 | 106 | 33.0 % | 43.4 % | +10.4 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 37.8 %, with 44.5 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 6.4 % | 6.6 % | +0.2 | [-2.8, +2.9] |
| R2 | 612 | 4.2 % | 3.4 % | -0.8 | [-3.2, +0.7] |
| R3 | 589 | 4.8 % | 4.8 % | +0.0 | [-1.2, +1.4] |
| R4 | 493 | 6.1 % | 4.1 % | -2.0 | [-3.7, -0.6] |
| R5 | 548 | 2.9 % | 5.4 % | +2.5 | [+1.2, +3.8] |
| R6 | 28 | 10.7 % | 14.3 % | +3.6 | [-7.7, +17.9] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 58.4 % | 63.4 % | +5.0 | [+0.6, +9.9] |
| threat faceswap | 272 | 45.2 % | 51.5 % | +6.2 | [+3.4, +9.6] |
| threat gan_face | 153 | 43.8 % | 47.1 % | +3.3 | [-0.0, +7.2] |
| threat personalised | 115 | 90.4 % | 91.3 % | +0.9 | [+0.0, +2.6] |
| threat t2i | 1600 | 49.7 % | 60.4 % | +10.7 | [+9.0, +12.5] |
| family faceswap-stills | 272 | 45.2 % | 51.5 % | +6.2 | [+3.4, +9.6] |
| family flux1 | 85 | 49.4 % | 64.7 % | +15.3 | [+7.1, +23.5] |
| family flux2 | 70 | 24.3 % | 32.9 % | +8.6 | [+2.8, +15.5] |
| family gan-faces | 153 | 43.8 % | 47.1 % | +3.3 | [-0.0, +7.2] |
| family gpt-4o-image-1 | 41 | 0.0 % | 14.6 % | +14.6 | [+4.9, +26.8] |
| family gpt-image-1.5-2 | 57 | 10.5 % | 8.8 % | -1.8 | [-7.0, +3.5] |
| family hidream | 50 | 72.0 % | 82.0 % | +10.0 | [+0.0, +20.4] |
| family hunyuanimage | 24 | 95.8 % | 100.0 % | +4.2 | [+0.0, +12.5] |
| family ideogram | 47 | 51.1 % | 68.1 % | +17.0 | [+6.7, +29.5] |
| family imagen3 | 53 | 37.7 % | 47.2 % | +9.4 | [-3.8, +22.6] |
| family imagen4 | 106 | 29.2 % | 41.5 % | +12.3 | [+5.9, +19.1] |
| family legacy | 145 | 93.1 % | 94.5 % | +1.4 | [-1.3, +4.5] |
| family long-tail | 186 | 54.8 % | 63.4 % | +8.6 | [+3.4, +14.4] |
| family midjourney | 153 | 68.6 % | 81.0 % | +12.4 | [+7.5, +17.3] |
| family modern-t2i-faces | 115 | 40.0 % | 46.1 % | +6.1 | [+2.6, +9.9] |
| family nano-banana | 118 | 27.1 % | 35.6 % | +8.5 | [+3.4, +14.5] |
| family partial-edits | 161 | 58.4 % | 63.4 % | +5.0 | [+0.6, +9.9] |
| family personalised | 115 | 90.4 % | 91.3 % | +0.9 | [+0.0, +2.6] |
| family qwen-image | 106 | 27.4 % | 37.7 % | +10.4 | [+2.9, +17.6] |
| family recraft | 54 | 50.0 % | 75.9 % | +25.9 | [+13.5, +39.7] |
| family sd3 | 75 | 68.0 % | 96.0 % | +28.0 | [+18.1, +38.5] |
| family seedream3 | 35 | 54.3 % | 80.0 % | +25.7 | [+11.4, +41.2] |
| family seedream4 | 80 | 62.5 % | 71.2 % | +8.8 | [+1.2, +17.7] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
