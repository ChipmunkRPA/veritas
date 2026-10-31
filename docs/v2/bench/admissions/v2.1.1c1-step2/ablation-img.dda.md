# Ablation: img.dda

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7,img.hf.ateeqq,img.hf.dima806 GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.dda ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dfdc_b7, img.hf.ateeqq, img.hf.dima806
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (MSCOCO train2017)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3
- Operating point (D-53): reals scoring exactly τ count as flagged with weight q (the rank rule's share); served rule (s ≥ τ) weighted FPR: without 5.0 % (q 0.165), with 5.0 % (q 0.317)

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +14.0 pts, CI [+11.1, +17.0] |
| A3b | pass | family faceswap-stills, family flux1, family flux2, family gpt-4o-image-1, family hidream, family ideogram, family imagen3, family imagen4, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family qwen-image, family recraft, family sd3, family seedream3, family seedream4, threat faceswap, threat t2i |
| A4 | fail | pooled reals: Δ FPR +0.2 pts, CI [-0.7, +1.0] (blocks with a lower bound above +1.0, D-53); blocking strata (n ≥ 150, lower bound > +2.0): R5; not established: R1, R2, R3, R4, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 69.3 % | 81.7 % | +12.4 |
| F2 | 70 | 24.3 % | 38.6 % | +14.3 |
| F3 | 57 | 10.5 % | 14.0 % | +3.5 |
| F4 | 118 | 28.8 % | 40.7 % | +11.9 |
| F5 | 80 | 63.7 % | 77.5 % | +13.8 |
| F6 | 106 | 34.9 % | 45.3 % | +10.4 |
| F7 | 106 | 33.0 % | 65.1 % | +32.1 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 37.8 %, with 51.8 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 6.4 % | 3.8 % | -2.6 | [-4.0, +0.0] |
| R2 | 612 | 4.2 % | 3.6 % | -0.7 | [-3.0, +0.9] |
| R3 | 589 | 4.8 % | 5.4 % | +0.7 | [-0.7, +2.1] |
| R4 | 493 | 6.1 % | 4.5 % | -1.6 | [-3.2, +0.0] |
| R5 | 548 | 2.9 % | 7.1 % | +4.2 | [+2.4, +6.0] |
| R6 | 28 | 10.7 % | 10.7 % | +0.0 | [-15.4, +14.3] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 58.4 % | 65.8 % | +7.5 | [+2.5, +13.0] |
| threat faceswap | 272 | 45.2 % | 63.6 % | +18.4 | [+9.5, +30.9] |
| threat gan_face | 153 | 43.8 % | 51.6 % | +7.8 | [+3.3, +13.1] |
| threat personalised | 115 | 90.4 % | 94.8 % | +4.3 | [+0.9, +8.7] |
| threat t2i | 1600 | 49.7 % | 66.9 % | +17.3 | [+15.2, +19.3] |
| family faceswap-stills | 272 | 45.2 % | 63.6 % | +18.4 | [+9.5, +30.9] |
| family flux1 | 85 | 49.4 % | 67.1 % | +17.6 | [+8.4, +27.0] |
| family flux2 | 70 | 24.3 % | 38.6 % | +14.3 | [+6.8, +22.7] |
| family gan-faces | 153 | 43.8 % | 51.6 % | +7.8 | [+3.3, +13.1] |
| family gpt-4o-image-1 | 41 | 0.0 % | 22.0 % | +22.0 | [+9.8, +36.6] |
| family gpt-image-1.5-2 | 57 | 10.5 % | 14.0 % | +3.5 | [-3.5, +12.3] |
| family hidream | 50 | 72.0 % | 88.0 % | +16.0 | [+5.4, +28.3] |
| family hunyuanimage | 24 | 95.8 % | 100.0 % | +4.2 | [+0.0, +12.5] |
| family ideogram | 47 | 51.1 % | 72.3 % | +21.3 | [+10.4, +34.1] |
| family imagen3 | 53 | 37.7 % | 56.6 % | +18.9 | [+3.8, +32.1] |
| family imagen4 | 106 | 29.2 % | 61.3 % | +32.1 | [+22.8, +41.4] |
| family legacy | 145 | 93.1 % | 94.5 % | +1.4 | [-1.3, +4.5] |
| family long-tail | 186 | 54.8 % | 70.4 % | +15.6 | [+8.7, +22.3] |
| family midjourney | 153 | 68.6 % | 82.4 % | +13.7 | [+8.4, +19.1] |
| family modern-t2i-faces | 115 | 40.0 % | 67.8 % | +27.8 | [+19.3, +35.5] |
| family nano-banana | 118 | 27.1 % | 40.7 % | +13.6 | [+6.8, +20.5] |
| family partial-edits | 161 | 58.4 % | 65.8 % | +7.5 | [+2.5, +13.0] |
| family personalised | 115 | 90.4 % | 94.8 % | +4.3 | [+0.9, +8.7] |
| family qwen-image | 106 | 27.4 % | 40.6 % | +13.2 | [+4.8, +21.6] |
| family recraft | 54 | 50.0 % | 83.3 % | +33.3 | [+20.4, +48.1] |
| family sd3 | 75 | 68.0 % | 98.7 % | +30.7 | [+20.5, +41.1] |
| family seedream3 | 35 | 54.3 % | 82.9 % | +28.6 | [+14.3, +44.1] |
| family seedream4 | 80 | 62.5 % | 77.5 % | +15.0 | [+7.5, +23.7] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
