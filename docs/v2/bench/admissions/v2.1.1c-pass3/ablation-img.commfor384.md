# Ablation: img.commfor384

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=face.gend_clip_l14,img.commfor384 GATES='face.gend_clip_l14=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.commfor384 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: face.gend_clip_l14
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (OwensLab/CommunityForensics-Eval)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +27.3 pts, CI [+23.7, +30.8] |
| A3b | pass | family flux1, family gan-faces, family hidream, family hunyuanimage, family ideogram, family imagen3, family imagen4, family legacy, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family partial-edits, family personalised, family qwen-image, family recraft, family sd3, family seedream3, family seedream4, threat edit, threat gan_face, threat personalised, threat t2i |
| A4 | fail | pooled reals: Δ FPR +0.3 pts, CI [-1.8, +2.0] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): R2; not established: R1, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): R6 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 3.9 % | 68.0 % | +64.1 |
| F2 | 70 | 7.1 % | 17.1 % | +10.0 |
| F3 | 57 | 0.0 % | 7.0 % | +7.0 |
| F4 | 118 | 0.8 % | 19.5 % | +18.6 |
| F5 | 80 | 3.8 % | 63.7 % | +60.0 |
| F6 | 106 | 0.9 % | 18.9 % | +17.9 |
| F7 | 106 | 6.6 % | 19.8 % | +13.2 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 3.3 %, with 30.6 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 14.6 % | 8.7 % | -5.9 | [-12.4, +2.3] |
| R2 | 612 | 0.5 % | 4.4 % | +3.9 | [+2.2, +5.4] |
| R3 | 589 | 3.4 % | 4.2 % | +0.8 | [-1.2, +3.0] |
| R4 | 493 | 1.2 % | 2.4 % | +1.2 | [-0.2, +2.6] |
| R5 | 548 | 5.1 % | 4.7 % | -0.4 | [-2.0, +1.3] |
| R6 | 28 | 3.6 % | 7.1 % | +3.6 | [-9.4, +16.7] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 11.2 % | 57.1 % | +46.0 | [+37.3, +54.7] |
| threat faceswap | 272 | 74.3 % | 78.3 % | +4.0 | [-1.3, +9.9] |
| threat gan_face | 153 | 47.7 % | 82.4 % | +34.6 | [+27.5, +41.8] |
| threat personalised | 115 | 33.0 % | 94.8 % | +61.7 | [+53.0, +71.3] |
| threat t2i | 1600 | 3.1 % | 42.9 % | +39.9 | [+37.1, +42.5] |
| family faceswap-stills | 272 | 74.3 % | 78.3 % | +4.0 | [-1.3, +9.9] |
| family flux1 | 85 | 1.2 % | 36.5 % | +35.3 | [+25.6, +45.9] |
| family flux2 | 70 | 10.0 % | 15.7 % | +5.7 | [-6.2, +17.9] |
| family gan-faces | 153 | 47.7 % | 82.4 % | +34.6 | [+27.5, +41.8] |
| family gpt-4o-image-1 | 41 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| family gpt-image-1.5-2 | 57 | 0.0 % | 7.0 % | +7.0 | [+1.8, +14.0] |
| family hidream | 50 | 2.0 % | 54.0 % | +52.0 | [+37.5, +66.1] |
| family hunyuanimage | 24 | 0.0 % | 95.8 % | +95.8 | [+87.5, +100.0] |
| family ideogram | 47 | 0.0 % | 51.1 % | +51.1 | [+35.6, +65.4] |
| family imagen3 | 53 | 0.0 % | 20.8 % | +20.8 | [+9.4, +32.1] |
| family imagen4 | 106 | 3.8 % | 19.8 % | +16.0 | [+6.9, +25.0] |
| family legacy | 145 | 2.8 % | 77.9 % | +75.2 | [+68.7, +81.4] |
| family long-tail | 186 | 4.8 % | 49.5 % | +44.6 | [+35.9, +52.5] |
| family midjourney | 153 | 3.3 % | 66.7 % | +63.4 | [+55.1, +71.7] |
| family modern-t2i-faces | 115 | 6.1 % | 38.3 % | +32.2 | [+22.9, +41.7] |
| family nano-banana | 118 | 1.7 % | 19.5 % | +17.8 | [+10.1, +25.6] |
| family partial-edits | 161 | 11.2 % | 57.1 % | +46.0 | [+37.3, +54.7] |
| family personalised | 115 | 33.0 % | 94.8 % | +61.7 | [+53.0, +71.3] |
| family qwen-image | 106 | 0.9 % | 17.9 % | +17.0 | [+10.5, +24.0] |
| family recraft | 54 | 3.7 % | 46.3 % | +42.6 | [+26.9, +57.1] |
| family sd3 | 75 | 2.7 % | 68.0 % | +65.3 | [+53.3, +76.9] |
| family seedream3 | 35 | 5.7 % | 45.7 % | +40.0 | [+24.3, +58.8] |
| family seedream4 | 80 | 2.5 % | 62.5 % | +60.0 | [+48.1, +71.8] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
