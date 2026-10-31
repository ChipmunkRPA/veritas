# Ablation: img.commfor384

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.hf.ateeqq,img.commfor384 GATES='img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.commfor384 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (OwensLab/CommunityForensics-Eval)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -2.5 pts, CI [-6.6, +1.8] |
| A3b | fail | family faceswap-stills, family gan-faces, family hunyuanimage, family ideogram, family imagen3, family legacy, family long-tail, family midjourney, family modern-t2i-faces, family partial-edits, family personalised, family recraft, family sd3, family seedream3, family seedream4, threat edit, threat faceswap, threat gan_face, threat personalised, threat t2i |
| A4 | pass | pooled reals: Δ FPR -0.2 pts, CI [-1.5, +0.9] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R4; above +2.0 on fewer than 150 reals (no term): R6 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 37.3 % | 68.6 % | +31.4 |
| F2 | 70 | 42.9 % | 22.9 % | -20.0 |
| F3 | 57 | 40.4 % | 12.3 % | -28.1 |
| F4 | 118 | 49.2 % | 30.5 % | -18.6 |
| F5 | 80 | 40.0 % | 63.7 % | +23.7 |
| F6 | 106 | 34.9 % | 27.4 % | -7.5 |
| F7 | 106 | 32.1 % | 34.0 % | +1.9 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 39.5 %, with 37.0 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 4.5 % | 5.9 % | +1.4 | [-3.8, +3.9] |
| R2 | 612 | 4.2 % | 5.6 % | +1.3 | [-1.3, +3.3] |
| R3 | 589 | 10.0 % | 5.4 % | -4.6 | [-7.5, -1.7] |
| R4 | 493 | 4.9 % | 6.9 % | +2.0 | [+0.0, +4.1] |
| R5 | 548 | 1.5 % | 0.7 % | -0.7 | [-1.8, +0.2] |
| R6 | 28 | 7.1 % | 10.7 % | +3.6 | [-8.3, +16.7] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 16.8 % | 57.8 % | +41.0 | [+31.7, +50.3] |
| threat faceswap | 272 | 0.4 % | 15.8 % | +15.4 | [+10.5, +21.8] |
| threat gan_face | 153 | 0.0 % | 43.8 % | +43.8 | [+36.6, +51.6] |
| threat personalised | 115 | 9.6 % | 94.8 % | +85.2 | [+79.1, +91.3] |
| threat t2i | 1600 | 36.4 % | 50.7 % | +14.4 | [+11.0, +17.6] |
| family faceswap-stills | 272 | 0.4 % | 15.8 % | +15.4 | [+10.5, +21.8] |
| family flux1 | 85 | 58.8 % | 49.4 % | -9.4 | [-22.9, +3.5] |
| family flux2 | 70 | 42.9 % | 21.4 % | -21.4 | [-32.4, -10.0] |
| family gan-faces | 153 | 0.0 % | 43.8 % | +43.8 | [+36.6, +51.6] |
| family gpt-4o-image-1 | 41 | 41.5 % | 0.0 % | -41.5 | [-56.2, -26.8] |
| family gpt-image-1.5-2 | 57 | 40.4 % | 12.3 % | -28.1 | [-43.9, -12.3] |
| family hidream | 50 | 62.0 % | 76.0 % | +14.0 | [-5.7, +32.7] |
| family hunyuanimage | 24 | 62.5 % | 95.8 % | +33.3 | [+12.5, +58.3] |
| family ideogram | 47 | 27.7 % | 57.4 % | +29.8 | [+10.9, +47.9] |
| family imagen3 | 53 | 18.9 % | 35.8 % | +17.0 | [+3.8, +30.2] |
| family imagen4 | 106 | 31.1 % | 33.0 % | +1.9 | [-8.3, +11.5] |
| family legacy | 145 | 39.3 % | 93.1 % | +53.8 | [+45.8, +61.8] |
| family long-tail | 186 | 40.3 % | 57.0 % | +16.7 | [+6.7, +26.3] |
| family midjourney | 153 | 35.9 % | 69.3 % | +33.3 | [+24.1, +42.7] |
| family modern-t2i-faces | 115 | 2.6 % | 36.5 % | +33.9 | [+25.9, +42.1] |
| family nano-banana | 118 | 48.3 % | 32.2 % | -16.1 | [-25.0, -6.8] |
| family partial-edits | 161 | 16.8 % | 57.8 % | +41.0 | [+31.7, +50.3] |
| family personalised | 115 | 9.6 % | 94.8 % | +85.2 | [+79.1, +91.3] |
| family qwen-image | 106 | 33.0 % | 27.4 % | -5.7 | [-15.2, +3.8] |
| family recraft | 54 | 27.8 % | 53.7 % | +25.9 | [+11.1, +40.7] |
| family sd3 | 75 | 29.3 % | 69.3 % | +40.0 | [+27.0, +52.9] |
| family seedream3 | 35 | 28.6 % | 54.3 % | +25.7 | [+11.4, +41.2] |
| family seedream4 | 80 | 38.8 % | 62.5 % | +23.8 | [+12.2, +35.4] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
