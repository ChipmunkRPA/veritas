# Ablation: img.veritas_probe.pe_core_l14

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=face.gend_clip_l14,img.veritas_probe.pe_core_l14 GATES='face.gend_clip_l14=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.veritas_probe.pe_core_l14 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: face.gend_clip_l14
- Items (fit ∪ cal): 5001 (2697 real, 2304 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +61.0 pts, CI [+57.0, +64.9] |
| A3b | pass | family flux1, family flux2, family gpt-4o-image-1, family gpt-image-1.5-2, family hidream, family hunyuanimage, family ideogram, family imagen3, family imagen4, family legacy, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family partial-edits, family personalised, family qwen-image, family recraft, family sd3, family seedream3, family seedream4, threat edit, threat personalised, threat t2i |
| A4 | fail | pooled reals: Δ FPR +0.4 pts, CI [-2.2, +2.6] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): R3, R4; not established: R1, R2, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 3.9 % | 82.4 % | +78.4 |
| F2 | 70 | 7.1 % | 52.9 % | +45.7 |
| F3 | 57 | 0.0 % | 47.4 % | +47.4 |
| F4 | 118 | 0.8 % | 68.6 % | +67.8 |
| F5 | 80 | 3.8 % | 46.2 % | +42.5 |
| F6 | 106 | 0.9 % | 78.3 % | +77.4 |
| F7 | 106 | 6.6 % | 74.5 % | +67.9 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 3.3 %, with 64.3 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 12.0 % | 2.1 % | -9.9 | [-18.5, +0.5] |
| R2 | 612 | 0.5 % | 0.0 % | -0.5 | [-1.2, +0.0] |
| R3 | 589 | 5.3 % | 11.2 % | +5.9 | [+3.0, +8.7] |
| R4 | 495 | 1.6 % | 9.9 % | +8.3 | [+5.5, +11.1] |
| R5 | 549 | 5.5 % | 2.0 % | -3.5 | [-5.6, -1.5] |
| R6 | 28 | 3.6 % | 0.0 % | -3.6 | [-12.5, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 11.8 % | 42.2 % | +30.4 | [+23.0, +37.9] |
| threat faceswap | 272 | 78.3 % | 41.9 % | -36.4 | [-46.9, -27.5] |
| threat gan_face | 153 | 54.2 % | 9.2 % | -45.1 | [-52.9, -37.3] |
| threat personalised | 115 | 40.9 % | 60.0 % | +19.1 | [+9.6, +28.7] |
| threat t2i | 1603 | 3.9 % | 74.8 % | +70.9 | [+68.2, +73.5] |
| family faceswap-stills | 272 | 78.3 % | 41.9 % | -36.4 | [-46.9, -27.5] |
| family flux1 | 85 | 1.2 % | 56.5 % | +55.3 | [+44.2, +66.3] |
| family flux2 | 70 | 5.7 % | 54.3 % | +48.6 | [+34.8, +61.4] |
| family gan-faces | 153 | 54.2 % | 9.2 % | -45.1 | [-52.9, -37.3] |
| family gpt-4o-image-1 | 41 | 2.4 % | 80.5 % | +78.0 | [+65.9, +90.2] |
| family gpt-image-1.5-2 | 57 | 0.0 % | 54.4 % | +54.4 | [+42.1, +66.7] |
| family hidream | 50 | 0.0 % | 90.0 % | +90.0 | [+81.2, +97.9] |
| family hunyuanimage | 24 | 0.0 % | 91.7 % | +91.7 | [+79.2, +100.0] |
| family ideogram | 47 | 2.1 % | 91.5 % | +89.4 | [+78.7, +97.8] |
| family imagen3 | 53 | 0.0 % | 67.9 % | +67.9 | [+54.7, +81.1] |
| family imagen4 | 106 | 7.5 % | 77.4 % | +69.8 | [+60.9, +77.7] |
| family legacy | 148 | 2.7 % | 54.1 % | +51.4 | [+42.1, +60.8] |
| family long-tail | 186 | 7.0 % | 76.9 % | +69.9 | [+62.4, +78.0] |
| family midjourney | 153 | 5.9 % | 89.5 % | +83.7 | [+77.2, +89.7] |
| family modern-t2i-faces | 115 | 7.0 % | 93.0 % | +86.1 | [+78.2, +93.1] |
| family nano-banana | 118 | 0.8 % | 78.0 % | +77.1 | [+69.2, +84.7] |
| family partial-edits | 161 | 11.8 % | 42.2 % | +30.4 | [+23.0, +37.9] |
| family personalised | 115 | 40.9 % | 60.0 % | +19.1 | [+9.6, +28.7] |
| family qwen-image | 106 | 0.9 % | 86.8 % | +85.8 | [+79.3, +92.0] |
| family recraft | 54 | 9.3 % | 87.0 % | +77.8 | [+65.4, +88.9] |
| family sd3 | 75 | 2.7 % | 57.3 % | +54.7 | [+41.9, +65.9] |
| family seedream3 | 35 | 5.7 % | 85.7 % | +80.0 | [+67.5, +91.4] |
| family seedream4 | 80 | 3.8 % | 62.5 % | +58.8 | [+48.1, +69.6] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
