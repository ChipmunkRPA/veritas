# Ablation: img.dda

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=face.gend_clip_l14,img.dda GATES='face.gend_clip_l14=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.dda ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: face.gend_clip_l14
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (MSCOCO train2017)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +44.8 pts, CI [+41.0, +48.7] |
| A3b | pass | family flux1, family flux2, family gan-faces, family gpt-4o-image-1, family gpt-image-1.5-2, family hidream, family hunyuanimage, family ideogram, family imagen3, family imagen4, family legacy, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family partial-edits, family personalised, family qwen-image, family recraft, family sd3, family seedream3, family seedream4, threat edit, threat gan_face, threat personalised, threat t2i |
| A4 | fail | pooled reals: Δ FPR +0.6 pts, CI [-2.8, +3.2] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R2, R4; above +2.0 on fewer than 150 reals (no term): R6 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 3.9 % | 82.4 % | +78.4 |
| F2 | 70 | 7.1 % | 32.9 % | +25.7 |
| F3 | 57 | 0.0 % | 17.5 % | +17.5 |
| F4 | 118 | 0.8 % | 44.9 % | +44.1 |
| F5 | 80 | 3.8 % | 65.0 % | +61.3 |
| F6 | 106 | 0.9 % | 29.2 % | +28.3 |
| F7 | 106 | 6.6 % | 65.1 % | +58.5 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 3.3 %, with 48.1 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 14.6 % | 3.8 % | -10.8 | [-20.1, -0.5] |
| R2 | 612 | 0.5 % | 6.9 % | +6.4 | [+1.3, +8.9] |
| R3 | 589 | 3.4 % | 4.8 % | +1.4 | [-0.5, +3.4] |
| R4 | 493 | 1.2 % | 4.5 % | +3.2 | [+1.4, +5.1] |
| R5 | 548 | 5.1 % | 4.6 % | -0.5 | [-2.0, +0.9] |
| R6 | 28 | 3.6 % | 10.7 % | +7.1 | [-7.1, +23.1] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 11.2 % | 65.8 % | +54.7 | [+46.0, +62.7] |
| threat faceswap | 272 | 74.3 % | 74.3 % | +0.0 | [-5.0, +4.9] |
| threat gan_face | 153 | 47.7 % | 58.8 % | +11.1 | [+5.9, +16.3] |
| threat personalised | 115 | 33.0 % | 95.7 % | +62.6 | [+53.9, +71.3] |
| threat t2i | 1600 | 3.1 % | 61.3 % | +58.2 | [+55.5, +60.8] |
| family faceswap-stills | 272 | 74.3 % | 74.3 % | +0.0 | [-5.0, +4.9] |
| family flux1 | 85 | 1.2 % | 61.2 % | +60.0 | [+50.6, +69.3] |
| family flux2 | 70 | 10.0 % | 30.0 % | +20.0 | [+7.0, +32.4] |
| family gan-faces | 153 | 47.7 % | 58.8 % | +11.1 | [+5.9, +16.3] |
| family gpt-4o-image-1 | 41 | 0.0 % | 31.7 % | +31.7 | [+17.1, +46.3] |
| family gpt-image-1.5-2 | 57 | 0.0 % | 15.8 % | +15.8 | [+7.0, +26.3] |
| family hidream | 50 | 2.0 % | 80.0 % | +78.0 | [+65.4, +89.6] |
| family hunyuanimage | 24 | 0.0 % | 100.0 % | +100.0 | [+100.0, +100.0] |
| family ideogram | 47 | 0.0 % | 72.3 % | +72.3 | [+57.8, +84.8] |
| family imagen3 | 53 | 0.0 % | 50.9 % | +50.9 | [+37.7, +64.2] |
| family imagen4 | 106 | 3.8 % | 55.7 % | +51.9 | [+42.0, +61.9] |
| family legacy | 145 | 2.8 % | 81.4 % | +78.6 | [+72.4, +84.6] |
| family long-tail | 186 | 4.8 % | 64.0 % | +59.1 | [+51.4, +66.7] |
| family midjourney | 153 | 3.3 % | 80.4 % | +77.1 | [+70.3, +83.7] |
| family modern-t2i-faces | 115 | 6.1 % | 67.0 % | +60.9 | [+50.4, +70.7] |
| family nano-banana | 118 | 1.7 % | 41.5 % | +39.8 | [+30.5, +49.2] |
| family partial-edits | 161 | 11.2 % | 65.8 % | +54.7 | [+46.0, +62.7] |
| family personalised | 115 | 33.0 % | 95.7 % | +62.6 | [+53.9, +71.3] |
| family qwen-image | 106 | 0.9 % | 26.4 % | +25.5 | [+17.0, +34.0] |
| family recraft | 54 | 3.7 % | 77.8 % | +74.1 | [+63.0, +85.2] |
| family sd3 | 75 | 2.7 % | 97.3 % | +94.7 | [+89.2, +98.7] |
| family seedream3 | 35 | 5.7 % | 74.3 % | +68.6 | [+51.3, +85.3] |
| family seedream4 | 80 | 2.5 % | 58.8 % | +56.2 | [+43.9, +68.3] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
