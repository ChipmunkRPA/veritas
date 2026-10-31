# Ablation: img.hf.haywoodsloan

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.hf.ateeqq,img.hf.haywoodsloan GATES='img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.haywoodsloan ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.hf.ateeqq
- Items (fit ∪ cal): 5001 (2697 real, 2304 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +31.4 pts, CI [+26.1, +36.5] |
| A3b | pass | family gan-faces, family gpt-4o-image-1, family hidream, family ideogram, family imagen3, family imagen4, family legacy, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family partial-edits, family personalised, family qwen-image, family recraft, family sd3, family seedream3, family seedream4, threat edit, threat gan_face, threat personalised, threat t2i |
| A4 | fail | pooled reals: Δ FPR -0.2 pts, CI [-1.9, +1.7] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R1, R5; above +2.0 on fewer than 150 reals (no term): R6 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 35.9 % | 81.0 % | +45.1 |
| F2 | 70 | 42.9 % | 48.6 % | +5.7 |
| F3 | 57 | 40.4 % | 47.4 % | +7.0 |
| F4 | 118 | 49.2 % | 72.0 % | +22.9 |
| F5 | 80 | 40.0 % | 77.5 % | +37.5 |
| F6 | 106 | 34.9 % | 81.1 % | +46.2 |
| F7 | 106 | 32.1 % | 87.7 % | +55.7 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 39.3 %, with 70.8 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 5.4 % | 8.0 % | +2.6 | [-4.7, +9.2] |
| R2 | 612 | 3.9 % | 4.2 % | +0.3 | [-2.0, +3.0] |
| R3 | 589 | 10.4 % | 6.1 % | -4.2 | [-7.1, -1.4] |
| R4 | 495 | 3.8 % | 1.4 % | -2.4 | [-4.4, -0.4] |
| R5 | 549 | 1.5 % | 4.7 % | +3.3 | [+1.3, +5.3] |
| R6 | 28 | 7.1 % | 10.7 % | +3.6 | [-12.5, +20.8] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 15.5 % | 36.0 % | +20.5 | [+13.0, +28.6] |
| threat faceswap | 272 | 0.4 % | 3.3 % | +2.9 | [+0.8, +5.6] |
| threat gan_face | 153 | 0.0 % | 15.7 % | +15.7 | [+10.5, +21.6] |
| threat personalised | 115 | 9.6 % | 55.7 % | +46.1 | [+36.5, +54.8] |
| threat t2i | 1603 | 36.1 % | 72.8 % | +36.7 | [+33.1, +40.6] |
| family faceswap-stills | 272 | 0.4 % | 3.3 % | +2.9 | [+0.8, +5.6] |
| family flux1 | 85 | 61.2 % | 65.9 % | +4.7 | [-8.1, +17.2] |
| family flux2 | 70 | 41.4 % | 45.7 % | +4.3 | [-9.7, +18.6] |
| family gan-faces | 153 | 0.0 % | 15.7 % | +15.7 | [+10.5, +21.6] |
| family gpt-4o-image-1 | 41 | 36.6 % | 75.6 % | +39.0 | [+17.1, +56.1] |
| family gpt-image-1.5-2 | 57 | 40.4 % | 45.6 % | +5.3 | [-14.0, +24.6] |
| family hidream | 50 | 60.0 % | 84.0 % | +24.0 | [+6.0, +42.6] |
| family hunyuanimage | 24 | 62.5 % | 87.5 % | +25.0 | [+0.0, +50.0] |
| family ideogram | 47 | 27.7 % | 85.1 % | +57.4 | [+40.4, +72.9] |
| family imagen3 | 53 | 18.9 % | 54.7 % | +35.8 | [+22.6, +49.1] |
| family imagen4 | 106 | 30.2 % | 86.8 % | +56.6 | [+44.1, +68.5] |
| family legacy | 148 | 37.8 % | 55.4 % | +17.6 | [+8.9, +25.7] |
| family long-tail | 186 | 40.3 % | 80.6 % | +40.3 | [+29.6, +50.2] |
| family midjourney | 153 | 34.6 % | 80.4 % | +45.8 | [+36.1, +55.3] |
| family modern-t2i-faces | 115 | 2.6 % | 98.3 % | +95.7 | [+91.8, +98.4] |
| family nano-banana | 118 | 47.5 % | 69.5 % | +22.0 | [+11.8, +32.5] |
| family partial-edits | 161 | 15.5 % | 36.0 % | +20.5 | [+13.0, +28.6] |
| family personalised | 115 | 9.6 % | 55.7 % | +46.1 | [+36.5, +54.8] |
| family qwen-image | 106 | 34.0 % | 79.2 % | +45.3 | [+33.3, +56.9] |
| family recraft | 54 | 27.8 % | 83.3 % | +55.6 | [+41.1, +69.2] |
| family sd3 | 75 | 30.7 % | 46.7 % | +16.0 | [+4.1, +27.3] |
| family seedream3 | 35 | 28.6 % | 60.0 % | +31.4 | [+17.1, +47.1] |
| family seedream4 | 80 | 40.0 % | 78.8 % | +38.7 | [+26.2, +50.6] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
