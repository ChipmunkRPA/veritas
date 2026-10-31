# Ablation: img.dda

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7,img.hf.ateeqq,img.hf.dima806 GATES='img.dfdc_b7=has_face;img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.dda ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dfdc_b7, img.hf.ateeqq, img.hf.dima806
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (MSCOCO train2017)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3
- Operating point (D-53): reals scoring exactly τ count as flagged with weight q (the rank rule's share); served rule (s ≥ τ) weighted FPR: without 5.0 % (q 0.185), with 5.0 % (q 0.365)

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +16.4 pts, CI [+13.2, +19.6] |
| A3b | pass | family flux1, family flux2, family gpt-4o-image-1, family hidream, family ideogram, family imagen3, family imagen4, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family qwen-image, family recraft, family sd3, family seedream3, family seedream4, threat t2i |
| A4 | pass | pooled reals: Δ FPR +0.1 pts, CI [-0.7, +0.8] (blocks with a lower bound above +1.0, D-53); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 69.3 % | 87.6 % | +18.3 |
| F2 | 70 | 25.7 % | 38.6 % | +12.9 |
| F3 | 57 | 12.3 % | 19.3 % | +7.0 |
| F4 | 118 | 30.5 % | 46.6 % | +16.1 |
| F5 | 80 | 65.0 % | 81.2 % | +16.2 |
| F6 | 106 | 33.0 % | 44.3 % | +11.3 |
| F7 | 106 | 33.0 % | 66.0 % | +33.0 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 38.4 %, with 54.8 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 6.2 % | 4.5 % | -1.7 | [-3.4, +0.6] |
| R2 | 612 | 5.4 % | 5.2 % | -0.2 | [-2.7, +1.4] |
| R3 | 589 | 5.1 % | 5.8 % | +0.7 | [-0.8, +2.3] |
| R4 | 493 | 6.7 % | 7.2 % | +0.5 | [-1.6, +2.6] |
| R5 | 548 | 0.9 % | 1.6 % | +0.7 | [+0.0, +1.6] |
| R6 | 28 | 14.3 % | 14.3 % | +0.0 | [-10.0, +10.7] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 59.0 % | 65.8 % | +6.8 | [+2.5, +11.8] |
| threat faceswap | 272 | 65.1 % | 68.4 % | +3.3 | [+1.1, +5.9] |
| threat gan_face | 153 | 55.6 % | 62.1 % | +6.5 | [+2.6, +11.1] |
| threat personalised | 115 | 93.0 % | 97.4 % | +4.3 | [+0.9, +7.8] |
| threat t2i | 1600 | 50.7 % | 67.8 % | +17.1 | [+15.1, +19.1] |
| family faceswap-stills | 272 | 65.1 % | 68.4 % | +3.3 | [+1.1, +5.9] |
| family flux1 | 85 | 49.4 % | 69.4 % | +20.0 | [+10.8, +29.4] |
| family flux2 | 70 | 24.3 % | 35.7 % | +11.4 | [+2.9, +20.0] |
| family gan-faces | 153 | 55.6 % | 62.1 % | +6.5 | [+2.6, +11.1] |
| family gpt-4o-image-1 | 41 | 0.0 % | 26.8 % | +26.8 | [+14.6, +41.5] |
| family gpt-image-1.5-2 | 57 | 12.3 % | 15.8 % | +3.5 | [-3.5, +12.3] |
| family hidream | 50 | 78.0 % | 90.0 % | +12.0 | [+4.0, +22.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 55.3 % | 76.6 % | +21.3 | [+10.2, +33.4] |
| family imagen3 | 53 | 35.8 % | 54.7 % | +18.9 | [+3.8, +32.1] |
| family imagen4 | 106 | 30.2 % | 62.3 % | +32.1 | [+22.9, +41.7] |
| family legacy | 145 | 92.4 % | 95.9 % | +3.4 | [+0.7, +6.8] |
| family long-tail | 186 | 57.0 % | 71.0 % | +14.0 | [+7.4, +20.8] |
| family midjourney | 153 | 69.3 % | 85.0 % | +15.7 | [+10.3, +21.7] |
| family modern-t2i-faces | 115 | 35.7 % | 64.3 % | +28.7 | [+21.3, +36.2] |
| family nano-banana | 118 | 28.8 % | 41.5 % | +12.7 | [+6.8, +18.8] |
| family partial-edits | 161 | 59.0 % | 65.8 % | +6.8 | [+2.5, +11.8] |
| family personalised | 115 | 93.0 % | 97.4 % | +4.3 | [+0.9, +7.8] |
| family qwen-image | 106 | 29.2 % | 39.6 % | +10.4 | [+2.9, +17.4] |
| family recraft | 54 | 53.7 % | 92.6 % | +38.9 | [+25.0, +53.8] |
| family sd3 | 75 | 72.0 % | 98.7 % | +26.7 | [+16.2, +37.5] |
| family seedream3 | 35 | 57.1 % | 82.9 % | +25.7 | [+11.4, +41.2] |
| family seedream4 | 80 | 63.7 % | 77.5 % | +13.8 | [+5.1, +23.2] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
