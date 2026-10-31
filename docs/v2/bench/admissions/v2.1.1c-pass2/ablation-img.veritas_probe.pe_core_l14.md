# Ablation: img.veritas_probe.pe_core_l14

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.hf.dima806,face.gend_clip_l14,img.hf.ateeqq,img.veritas_probe.pe_core_l14 GATES='face.gend_clip_l14=has_face;img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.veritas_probe.pe_core_l14 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: face.gend_clip_l14, img.dda, img.hf.ateeqq, img.hf.dima806
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +18.4 pts, CI [+14.9, +21.8] |
| A3b | pass | family flux1, family flux2, family gpt-4o-image-1, family gpt-image-1.5-2, family hidream, family ideogram, family imagen3, family imagen4, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family qwen-image, family recraft, family seedream3, family seedream4, threat t2i |
| A4 | fail | pooled reals: Δ FPR +0.1 pts, CI [-1.1, +1.5] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R3; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 83.0 % | 94.1 % | +11.1 |
| F2 | 70 | 44.3 % | 52.9 % | +8.6 |
| F3 | 57 | 21.1 % | 40.4 % | +19.3 |
| F4 | 118 | 50.8 % | 76.3 % | +25.4 |
| F5 | 80 | 68.8 % | 83.8 % | +15.0 |
| F6 | 106 | 40.6 % | 71.7 % | +31.1 |
| F7 | 106 | 67.9 % | 85.8 % | +17.9 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 53.8 %, with 72.1 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 4.5 % | 1.9 % | -2.6 | [-4.6, +0.5] |
| R2 | 612 | 3.9 % | 0.0 % | -3.9 | [-5.8, -1.3] |
| R3 | 589 | 5.8 % | 9.8 % | +4.1 | [+1.7, +6.4] |
| R4 | 493 | 5.9 % | 7.5 % | +1.6 | [-1.0, +4.3] |
| R5 | 548 | 4.2 % | 6.0 % | +1.8 | [-0.2, +4.0] |
| R6 | 28 | 17.9 % | 0.0 % | -17.9 | [-34.6, -4.2] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 65.2 % | 69.6 % | +4.3 | [-1.2, +9.9] |
| threat faceswap | 272 | 71.3 % | 67.3 % | -4.0 | [-9.0, +0.8] |
| threat gan_face | 153 | 54.2 % | 44.4 % | -9.8 | [-17.0, -3.3] |
| threat personalised | 115 | 96.5 % | 97.4 % | +0.9 | [-2.6, +4.3] |
| threat t2i | 1600 | 65.1 % | 85.6 % | +20.6 | [+18.4, +22.8] |
| family faceswap-stills | 272 | 71.3 % | 67.3 % | -4.0 | [-9.0, +0.8] |
| family flux1 | 85 | 62.4 % | 78.8 % | +16.5 | [+8.3, +25.0] |
| family flux2 | 70 | 40.0 % | 54.3 % | +14.3 | [+5.9, +22.9] |
| family gan-faces | 153 | 54.2 % | 44.4 % | -9.8 | [-17.0, -3.3] |
| family gpt-4o-image-1 | 41 | 36.6 % | 82.9 % | +46.3 | [+31.7, +63.4] |
| family gpt-image-1.5-2 | 57 | 17.5 % | 47.4 % | +29.8 | [+17.5, +42.1] |
| family hidream | 50 | 84.0 % | 96.0 % | +12.0 | [+2.1, +22.9] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 76.6 % | 95.7 % | +19.1 | [+8.3, +31.4] |
| family imagen3 | 53 | 49.1 % | 81.1 % | +32.1 | [+17.0, +47.2] |
| family imagen4 | 106 | 62.3 % | 91.5 % | +29.2 | [+20.0, +39.0] |
| family legacy | 145 | 91.0 % | 91.7 % | +0.7 | [-3.8, +5.2] |
| family long-tail | 186 | 71.0 % | 88.2 % | +17.2 | [+12.0, +23.2] |
| family midjourney | 153 | 80.4 % | 95.4 % | +15.0 | [+9.4, +20.9] |
| family modern-t2i-faces | 115 | 66.1 % | 93.9 % | +27.8 | [+19.8, +36.0] |
| family nano-banana | 118 | 42.4 % | 78.8 % | +36.4 | [+28.0, +45.3] |
| family partial-edits | 161 | 65.2 % | 69.6 % | +4.3 | [-1.2, +9.9] |
| family personalised | 115 | 96.5 % | 97.4 % | +0.9 | [-2.6, +4.3] |
| family qwen-image | 106 | 33.0 % | 73.6 % | +40.6 | [+30.5, +50.9] |
| family recraft | 54 | 83.3 % | 94.4 % | +11.1 | [+3.6, +20.4] |
| family sd3 | 75 | 98.7 % | 97.3 % | -1.3 | [-6.3, +2.8] |
| family seedream3 | 35 | 80.0 % | 97.1 % | +17.1 | [+5.9, +30.6] |
| family seedream4 | 80 | 57.5 % | 83.8 % | +26.3 | [+16.0, +35.9] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
