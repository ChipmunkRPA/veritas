# Ablation: img.hf.haywoodsloan

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.hf.ateeqq,img.hf.haywoodsloan CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.haywoodsloan ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda, img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +18.6 pts, CI [+15.4, +22.0] |
| A3b | pass | family flux1, family gpt-4o-image-1, family gpt-image-1.5-2, family ideogram, family imagen3, family imagen4, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family qwen-image, threat t2i |
| A4 | fail | pooled reals: Δ FPR +0.7 pts, CI [-0.2, +1.6] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): R5; not established: R1, R2, R3, R4, R6; point above +2.0 without established harm: none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 83.7 % | 95.4 % | +11.8 |
| F2 | 70 | 37.1 % | 47.1 % | +10.0 |
| F3 | 57 | 15.8 % | 42.1 % | +26.3 |
| F4 | 118 | 48.3 % | 69.5 % | +21.2 |
| F5 | 80 | 76.2 % | 85.0 % | +8.8 |
| F6 | 106 | 39.6 % | 71.7 % | +32.1 |
| F7 | 106 | 66.0 % | 85.8 % | +19.8 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 52.4 %, with 71.0 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 4.2 % | 5.0 % | +0.7 | [-2.3, +2.6] |
| R2 | 612 | 3.8 % | 2.3 % | -1.5 | [-3.2, +0.3] |
| R3 | 589 | 4.6 % | 4.8 % | +0.2 | [-1.3, +1.5] |
| R4 | 493 | 4.5 % | 3.0 % | -1.4 | [-3.0, +0.2] |
| R5 | 548 | 2.0 % | 7.7 % | +5.7 | [+3.3, +8.0] |
| R6 | 28 | 10.7 % | 7.1 % | -3.6 | [-16.7, +7.7] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 65.2 % | 67.7 % | +2.5 | [-0.6, +6.2] |
| threat faceswap | 272 | 20.6 % | 16.9 % | -3.7 | [-6.6, -1.4] |
| threat gan_face | 153 | 43.8 % | 47.7 % | +3.9 | [+0.7, +7.8] |
| threat personalised | 115 | 96.5 % | 95.7 % | -0.9 | [-3.5, +1.7] |
| threat t2i | 1600 | 67.6 % | 80.3 % | +12.7 | [+10.9, +14.8] |
| family faceswap-stills | 272 | 20.6 % | 16.9 % | -3.7 | [-6.6, -1.4] |
| family flux1 | 85 | 67.1 % | 78.8 % | +11.8 | [+3.6, +20.9] |
| family flux2 | 70 | 37.1 % | 47.1 % | +10.0 | [+2.9, +18.3] |
| family gan-faces | 153 | 43.8 % | 47.7 % | +3.9 | [+0.7, +7.8] |
| family gpt-4o-image-1 | 41 | 22.0 % | 58.5 % | +36.6 | [+22.0, +53.7] |
| family gpt-image-1.5-2 | 57 | 17.5 % | 38.6 % | +21.1 | [+8.8, +31.6] |
| family hidream | 50 | 86.0 % | 90.0 % | +4.0 | [-6.1, +15.7] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 74.5 % | 87.2 % | +12.8 | [+4.2, +23.4] |
| family imagen3 | 53 | 52.8 % | 69.8 % | +17.0 | [+5.7, +30.2] |
| family imagen4 | 106 | 65.1 % | 83.0 % | +17.9 | [+9.3, +27.2] |
| family legacy | 145 | 95.2 % | 90.3 % | -4.8 | [-9.2, -0.7] |
| family long-tail | 186 | 69.9 % | 82.8 % | +12.9 | [+6.8, +19.3] |
| family midjourney | 153 | 83.7 % | 94.1 % | +10.5 | [+5.2, +16.0] |
| family modern-t2i-faces | 115 | 72.2 % | 93.0 % | +20.9 | [+14.2, +28.0] |
| family nano-banana | 118 | 44.1 % | 66.9 % | +22.9 | [+13.7, +31.6] |
| family partial-edits | 161 | 65.2 % | 67.7 % | +2.5 | [-0.6, +6.2] |
| family personalised | 115 | 96.5 % | 95.7 % | -0.9 | [-3.5, +1.7] |
| family qwen-image | 106 | 37.7 % | 68.9 % | +31.1 | [+21.0, +41.2] |
| family recraft | 54 | 85.2 % | 92.6 % | +7.4 | [+0.0, +16.7] |
| family sd3 | 75 | 98.7 % | 97.3 % | -1.3 | [-3.9, +0.0] |
| family seedream3 | 35 | 82.9 % | 80.0 % | -2.9 | [-8.8, +0.0] |
| family seedream4 | 80 | 75.0 % | 81.2 % | +6.2 | [+0.0, +12.8] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
