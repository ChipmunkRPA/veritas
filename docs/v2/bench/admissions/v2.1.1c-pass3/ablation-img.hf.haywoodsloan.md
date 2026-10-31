# Ablation: img.hf.haywoodsloan

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=face.gend_clip_l14,img.hf.haywoodsloan GATES='face.gend_clip_l14=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.haywoodsloan ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: face.gend_clip_l14
- Items (fit ∪ cal): 5001 (2697 real, 2304 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +54.3 pts, CI [+50.1, +58.3] |
| A3b | pass | family flux1, family flux2, family gpt-4o-image-1, family gpt-image-1.5-2, family hidream, family hunyuanimage, family ideogram, family imagen3, family imagen4, family legacy, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family partial-edits, family personalised, family qwen-image, family recraft, family sd3, family seedream3, family seedream4, threat edit, threat personalised, threat t2i |
| A4 | fail | pooled reals: Δ FPR +0.2 pts, CI [-0.8, +1.2] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): R2; not established: R1, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): R6 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 3.9 % | 77.1 % | +73.2 |
| F2 | 70 | 7.1 % | 42.9 % | +35.7 |
| F3 | 57 | 0.0 % | 28.1 % | +28.1 |
| F4 | 118 | 0.8 % | 61.0 % | +60.2 |
| F5 | 80 | 3.8 % | 68.8 % | +65.0 |
| F6 | 106 | 0.9 % | 60.4 % | +59.4 |
| F7 | 106 | 6.6 % | 65.1 % | +58.5 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 3.3 %, with 57.6 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 12.0 % | 9.2 % | -2.8 | [-6.9, +0.0] |
| R2 | 612 | 0.5 % | 4.9 % | +4.4 | [+2.7, +8.0] |
| R3 | 589 | 5.3 % | 4.8 % | -0.5 | [-2.8, +1.8] |
| R4 | 495 | 1.6 % | 2.2 % | +0.6 | [-1.0, +2.2] |
| R5 | 549 | 5.5 % | 3.5 % | -2.0 | [-4.4, +0.2] |
| R6 | 28 | 3.6 % | 10.7 % | +7.1 | [+0.0, +19.2] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 11.8 % | 32.9 % | +21.1 | [+13.7, +29.2] |
| threat faceswap | 272 | 78.3 % | 30.9 % | -47.4 | [-57.3, -37.6] |
| threat gan_face | 153 | 54.2 % | 28.8 % | -25.5 | [-34.0, -17.0] |
| threat personalised | 115 | 40.9 % | 60.0 % | +19.1 | [+8.7, +28.7] |
| threat t2i | 1603 | 3.9 % | 62.6 % | +58.7 | [+55.9, +61.4] |
| family faceswap-stills | 272 | 78.3 % | 30.9 % | -47.4 | [-57.3, -37.6] |
| family flux1 | 85 | 1.2 % | 56.5 % | +55.3 | [+44.6, +65.9] |
| family flux2 | 70 | 5.7 % | 44.3 % | +38.6 | [+25.0, +51.5] |
| family gan-faces | 153 | 54.2 % | 28.8 % | -25.5 | [-34.0, -17.0] |
| family gpt-4o-image-1 | 41 | 2.4 % | 70.7 % | +68.3 | [+53.7, +80.5] |
| family gpt-image-1.5-2 | 57 | 0.0 % | 29.8 % | +29.8 | [+17.5, +42.1] |
| family hidream | 50 | 0.0 % | 70.0 % | +70.0 | [+57.7, +82.0] |
| family hunyuanimage | 24 | 0.0 % | 75.0 % | +75.0 | [+58.3, +91.7] |
| family ideogram | 47 | 2.1 % | 70.2 % | +68.1 | [+54.3, +81.2] |
| family imagen3 | 53 | 0.0 % | 43.4 % | +43.4 | [+30.2, +56.6] |
| family imagen4 | 106 | 7.5 % | 67.0 % | +59.4 | [+50.0, +68.5] |
| family legacy | 148 | 2.7 % | 54.7 % | +52.0 | [+41.9, +61.4] |
| family long-tail | 186 | 7.0 % | 72.0 % | +65.1 | [+57.8, +71.9] |
| family midjourney | 153 | 5.9 % | 75.8 % | +69.9 | [+62.4, +77.2] |
| family modern-t2i-faces | 115 | 7.0 % | 74.8 % | +67.8 | [+56.8, +78.5] |
| family nano-banana | 118 | 0.8 % | 60.2 % | +59.3 | [+50.4, +68.4] |
| family partial-edits | 161 | 11.8 % | 32.9 % | +21.1 | [+13.7, +29.2] |
| family personalised | 115 | 40.9 % | 60.0 % | +19.1 | [+8.7, +28.7] |
| family qwen-image | 106 | 0.9 % | 64.2 % | +63.2 | [+54.4, +72.2] |
| family recraft | 54 | 9.3 % | 74.1 % | +64.8 | [+51.9, +76.9] |
| family sd3 | 75 | 2.7 % | 42.7 % | +40.0 | [+27.9, +52.0] |
| family seedream3 | 35 | 5.7 % | 48.6 % | +42.9 | [+25.7, +60.0] |
| family seedream4 | 80 | 3.8 % | 67.5 % | +63.8 | [+52.6, +74.7] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
