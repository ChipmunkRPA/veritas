# Ablation: img.hf.ateeqq

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=face.gend_clip_l14,img.hf.ateeqq GATES='face.gend_clip_l14=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.ateeqq ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: face.gend_clip_l14
- Items (fit ∪ cal): 5001 (2697 real, 2304 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +28.9 pts, CI [+24.9, +33.2] |
| A3b | pass | family flux1, family flux2, family gpt-image-1.5-2, family hidream, family hunyuanimage, family ideogram, family imagen3, family imagen4, family legacy, family long-tail, family midjourney, family nano-banana, family qwen-image, family sd3, family seedream4, threat t2i |
| A4 | pass | pooled reals: Δ FPR +0.0 pts, CI [-0.9, +0.9] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R3; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 3.9 % | 23.5 % | +19.6 |
| F2 | 70 | 7.1 % | 41.4 % | +34.3 |
| F3 | 57 | 0.0 % | 26.3 % | +26.3 |
| F4 | 118 | 0.8 % | 39.8 % | +39.0 |
| F5 | 80 | 3.8 % | 35.0 % | +31.2 |
| F6 | 106 | 0.9 % | 29.2 % | +28.3 |
| F7 | 106 | 6.6 % | 30.2 % | +23.6 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 3.3 %, with 32.2 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 12.0 % | 11.6 % | -0.5 | [-5.2, +1.8] |
| R2 | 612 | 0.5 % | 1.1 % | +0.7 | [-0.4, +2.2] |
| R3 | 589 | 5.3 % | 7.6 % | +2.4 | [+0.2, +4.8] |
| R4 | 495 | 1.6 % | 3.0 % | +1.4 | [-0.4, +3.2] |
| R5 | 549 | 5.5 % | 1.6 % | -3.8 | [-5.8, -2.0] |
| R6 | 28 | 3.6 % | 0.0 % | -3.6 | [-12.5, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 11.8 % | 16.8 % | +5.0 | [+0.6, +9.9] |
| threat faceswap | 272 | 78.3 % | 31.6 % | -46.7 | [-57.3, -35.2] |
| threat gan_face | 153 | 54.2 % | 9.8 % | -44.4 | [-52.3, -37.3] |
| threat personalised | 115 | 40.9 % | 29.6 % | -11.3 | [-21.7, -1.7] |
| threat t2i | 1603 | 3.9 % | 22.0 % | +18.0 | [+15.6, +20.5] |
| family faceswap-stills | 272 | 78.3 % | 31.6 % | -46.7 | [-57.3, -35.2] |
| family flux1 | 85 | 1.2 % | 23.5 % | +22.4 | [+14.1, +31.7] |
| family flux2 | 70 | 5.7 % | 37.1 % | +31.4 | [+18.2, +44.0] |
| family gan-faces | 153 | 54.2 % | 9.8 % | -44.4 | [-52.3, -37.3] |
| family gpt-4o-image-1 | 41 | 2.4 % | 14.6 % | +12.2 | [+0.0, +24.4] |
| family gpt-image-1.5-2 | 57 | 0.0 % | 19.3 % | +19.3 | [+10.5, +29.8] |
| family hidream | 50 | 0.0 % | 40.0 % | +40.0 | [+24.0, +55.1] |
| family hunyuanimage | 24 | 0.0 % | 33.3 % | +33.3 | [+16.7, +54.2] |
| family ideogram | 47 | 2.1 % | 14.9 % | +12.8 | [+4.2, +23.4] |
| family imagen3 | 53 | 0.0 % | 11.3 % | +11.3 | [+3.8, +20.8] |
| family imagen4 | 106 | 7.5 % | 30.2 % | +22.6 | [+13.3, +32.4] |
| family legacy | 148 | 2.7 % | 23.0 % | +20.3 | [+12.3, +28.3] |
| family long-tail | 186 | 7.0 % | 28.5 % | +21.5 | [+14.9, +28.8] |
| family midjourney | 153 | 5.9 % | 18.3 % | +12.4 | [+5.7, +18.9] |
| family modern-t2i-faces | 115 | 7.0 % | 9.6 % | +2.6 | [-0.9, +6.8] |
| family nano-banana | 118 | 0.8 % | 25.4 % | +24.6 | [+16.8, +32.5] |
| family partial-edits | 161 | 11.8 % | 16.8 % | +5.0 | [+0.6, +9.9] |
| family personalised | 115 | 40.9 % | 29.6 % | -11.3 | [-21.7, -1.7] |
| family qwen-image | 106 | 0.9 % | 16.0 % | +15.1 | [+8.3, +22.7] |
| family recraft | 54 | 9.3 % | 16.7 % | +7.4 | [-1.9, +17.3] |
| family sd3 | 75 | 2.7 % | 16.0 % | +13.3 | [+4.2, +23.1] |
| family seedream3 | 35 | 5.7 % | 11.4 % | +5.7 | [-5.6, +17.6] |
| family seedream4 | 80 | 3.8 % | 22.5 % | +18.8 | [+8.9, +29.3] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
