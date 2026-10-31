# Ablation: img.dda

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.hf.dima806,face.gend_clip_l14,img.hf.ateeqq GATES='face.gend_clip_l14=has_face;img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.dda ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: face.gend_clip_l14, img.hf.ateeqq, img.hf.dima806
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (MSCOCO train2017)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +20.6 pts, CI [+16.2, +25.0] |
| A3b | pass | family faceswap-stills, family gan-faces, family hidream, family hunyuanimage, family ideogram, family imagen3, family imagen4, family legacy, family long-tail, family midjourney, family modern-t2i-faces, family partial-edits, family personalised, family recraft, family sd3, family seedream3, family seedream4, threat edit, threat faceswap, threat gan_face, threat personalised, threat t2i |
| A4 | fail | pooled reals: Δ FPR -0.1 pts, CI [-1.4, +1.0] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R5; above +2.0 on fewer than 150 reals (no term): R6 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 27.5 % | 83.0 % | +55.6 |
| F2 | 70 | 40.0 % | 44.3 % | +4.3 |
| F3 | 57 | 35.1 % | 21.1 % | -14.0 |
| F4 | 118 | 37.3 % | 50.8 % | +13.6 |
| F5 | 80 | 35.0 % | 68.8 % | +33.8 |
| F6 | 106 | 33.0 % | 40.6 % | +7.5 |
| F7 | 106 | 24.5 % | 67.9 % | +43.4 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 33.2 %, with 53.8 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 3.1 % | 4.5 % | +1.4 | [-2.3, +3.7] |
| R2 | 612 | 3.4 % | 3.9 % | +0.5 | [-3.5, +2.6] |
| R3 | 589 | 10.0 % | 5.8 % | -4.2 | [-6.8, -1.8] |
| R4 | 493 | 6.5 % | 5.9 % | -0.6 | [-3.0, +1.8] |
| R5 | 548 | 1.8 % | 4.2 % | +2.4 | [+0.5, +4.0] |
| R6 | 28 | 7.1 % | 17.9 % | +10.7 | [+0.0, +25.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 18.0 % | 65.2 % | +47.2 | [+37.3, +57.1] |
| threat faceswap | 272 | 47.1 % | 71.3 % | +24.3 | [+17.5, +31.7] |
| threat gan_face | 153 | 21.6 % | 54.2 % | +32.7 | [+25.5, +39.9] |
| threat personalised | 115 | 14.8 % | 96.5 % | +81.7 | [+74.8, +88.7] |
| threat t2i | 1600 | 33.7 % | 65.1 % | +31.4 | [+27.9, +34.7] |
| family faceswap-stills | 272 | 47.1 % | 71.3 % | +24.3 | [+17.5, +31.7] |
| family flux1 | 85 | 57.6 % | 62.4 % | +4.7 | [-8.5, +16.7] |
| family flux2 | 70 | 41.4 % | 40.0 % | -1.4 | [-12.8, +10.0] |
| family gan-faces | 153 | 21.6 % | 54.2 % | +32.7 | [+25.5, +39.9] |
| family gpt-4o-image-1 | 41 | 34.1 % | 36.6 % | +2.4 | [-17.1, +22.0] |
| family gpt-image-1.5-2 | 57 | 38.6 % | 17.5 % | -21.1 | [-36.9, -5.3] |
| family hidream | 50 | 58.0 % | 84.0 % | +26.0 | [+7.8, +44.7] |
| family hunyuanimage | 24 | 62.5 % | 100.0 % | +37.5 | [+16.7, +58.3] |
| family ideogram | 47 | 23.4 % | 76.6 % | +53.2 | [+37.8, +67.4] |
| family imagen3 | 53 | 18.9 % | 49.1 % | +30.2 | [+13.2, +45.3] |
| family imagen4 | 106 | 25.5 % | 62.3 % | +36.8 | [+25.7, +47.2] |
| family legacy | 145 | 38.6 % | 91.0 % | +52.4 | [+44.9, +60.0] |
| family long-tail | 186 | 39.2 % | 71.0 % | +31.7 | [+20.9, +42.2] |
| family midjourney | 153 | 30.7 % | 80.4 % | +49.7 | [+40.6, +58.9] |
| family modern-t2i-faces | 115 | 2.6 % | 66.1 % | +63.5 | [+54.5, +72.3] |
| family nano-banana | 118 | 40.7 % | 42.4 % | +1.7 | [-8.5, +11.8] |
| family partial-edits | 161 | 18.0 % | 65.2 % | +47.2 | [+37.3, +57.1] |
| family personalised | 115 | 14.8 % | 96.5 % | +81.7 | [+74.8, +88.7] |
| family qwen-image | 106 | 31.1 % | 33.0 % | +1.9 | [-7.0, +11.3] |
| family recraft | 54 | 25.9 % | 83.3 % | +57.4 | [+43.1, +71.2] |
| family sd3 | 75 | 26.7 % | 98.7 % | +72.0 | [+60.5, +82.7] |
| family seedream3 | 35 | 28.6 % | 80.0 % | +51.4 | [+35.1, +68.6] |
| family seedream4 | 80 | 36.2 % | 57.5 % | +21.2 | [+7.3, +35.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
