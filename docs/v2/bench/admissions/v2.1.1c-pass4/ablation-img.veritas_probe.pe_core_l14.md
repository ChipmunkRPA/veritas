# Ablation: img.veritas_probe.pe_core_l14

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.hf.ateeqq,img.veritas_probe.pe_core_l14 GATES='img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.veritas_probe.pe_core_l14 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.hf.ateeqq
- Items (fit ∪ cal): 5001 (2697 real, 2304 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +39.0 pts, CI [+34.1, +44.0] |
| A3b | pass | family flux2, family gpt-4o-image-1, family gpt-image-1.5-2, family hidream, family hunyuanimage, family ideogram, family imagen3, family imagen4, family legacy, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family partial-edits, family personalised, family qwen-image, family recraft, family sd3, family seedream3, family seedream4, threat edit, threat personalised, threat t2i |
| A4 | fail | pooled reals: Δ FPR +0.1 pts, CI [-1.2, +1.5] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R3, R4; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 35.9 % | 92.8 % | +56.9 |
| F2 | 70 | 42.9 % | 60.0 % | +17.1 |
| F3 | 57 | 40.4 % | 66.7 % | +26.3 |
| F4 | 118 | 49.2 % | 83.9 % | +34.7 |
| F5 | 80 | 40.0 % | 68.8 % | +28.7 |
| F6 | 106 | 34.9 % | 87.7 % | +52.8 |
| F7 | 106 | 32.1 % | 88.7 % | +56.6 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 39.3 %, with 78.4 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 5.4 % | 1.9 % | -3.5 | [-5.5, +1.1] |
| R2 | 612 | 3.9 % | 0.0 % | -3.9 | [-6.8, -2.5] |
| R3 | 589 | 10.4 % | 13.9 % | +3.6 | [+0.3, +6.9] |
| R4 | 495 | 3.8 % | 6.9 % | +3.0 | [+0.4, +5.9] |
| R5 | 549 | 1.5 % | 2.6 % | +1.1 | [-0.5, +2.7] |
| R6 | 28 | 7.1 % | 3.6 % | -3.6 | [-16.7, +7.7] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 15.5 % | 45.3 % | +29.8 | [+21.1, +37.9] |
| threat faceswap | 272 | 0.4 % | 2.9 % | +2.6 | [+1.0, +4.7] |
| threat gan_face | 153 | 0.0 % | 3.9 % | +3.9 | [+1.3, +7.2] |
| threat personalised | 115 | 9.6 % | 63.5 % | +53.9 | [+44.3, +63.5] |
| threat t2i | 1603 | 36.1 % | 82.3 % | +46.3 | [+42.9, +50.0] |
| family faceswap-stills | 272 | 0.4 % | 2.9 % | +2.6 | [+1.0, +4.7] |
| family flux1 | 85 | 61.2 % | 72.9 % | +11.8 | [-1.1, +24.1] |
| family flux2 | 70 | 41.4 % | 58.6 % | +17.1 | [+4.2, +30.6] |
| family gan-faces | 153 | 0.0 % | 3.9 % | +3.9 | [+1.3, +7.2] |
| family gpt-4o-image-1 | 41 | 36.6 % | 90.2 % | +53.7 | [+36.6, +68.3] |
| family gpt-image-1.5-2 | 57 | 40.4 % | 68.4 % | +28.1 | [+10.5, +43.9] |
| family hidream | 50 | 60.0 % | 98.0 % | +38.0 | [+24.5, +53.1] |
| family hunyuanimage | 24 | 62.5 % | 95.8 % | +33.3 | [+12.5, +58.3] |
| family ideogram | 47 | 27.7 % | 91.5 % | +63.8 | [+48.9, +77.8] |
| family imagen3 | 53 | 18.9 % | 73.6 % | +54.7 | [+39.6, +69.8] |
| family imagen4 | 106 | 30.2 % | 92.5 % | +62.3 | [+51.9, +73.0] |
| family legacy | 148 | 37.8 % | 58.1 % | +20.3 | [+10.7, +30.1] |
| family long-tail | 186 | 40.3 % | 85.5 % | +45.2 | [+34.5, +55.2] |
| family midjourney | 153 | 34.6 % | 93.5 % | +58.8 | [+50.6, +67.3] |
| family modern-t2i-faces | 115 | 2.6 % | 98.3 % | +95.7 | [+92.1, +98.9] |
| family nano-banana | 118 | 47.5 % | 84.7 % | +37.3 | [+28.0, +47.0] |
| family partial-edits | 161 | 15.5 % | 45.3 % | +29.8 | [+21.1, +37.9] |
| family personalised | 115 | 9.6 % | 63.5 % | +53.9 | [+44.3, +63.5] |
| family qwen-image | 106 | 34.0 % | 89.6 % | +55.7 | [+45.2, +65.7] |
| family recraft | 54 | 27.8 % | 90.7 % | +63.0 | [+48.3, +75.9] |
| family sd3 | 75 | 30.7 % | 68.0 % | +37.3 | [+26.4, +48.1] |
| family seedream3 | 35 | 28.6 % | 91.4 % | +62.9 | [+47.2, +79.4] |
| family seedream4 | 80 | 40.0 % | 76.2 % | +36.2 | [+24.7, +48.1] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
