# Ablation: face.gend_clip_l14

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=face.gend_clip_l14,img.dda,img.dfdc_b7,img.hf.ateeqq,img.hf.dima806 GATES='face.gend_clip_l14=has_face;img.dfdc_b7=has_face;img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=face.gend_clip_l14 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.dfdc_b7, img.hf.ateeqq, img.hf.dima806
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (Celeb-DF v3, DeepSpeak v1.1, DeepSpeak v2, FFIW, FaceForensics++)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3
- Operating point (D-53): reals scoring exactly τ count as flagged with weight q (the rank rule's share); served rule (s ≥ τ) weighted FPR: without 5.0 % (q 0.407), with 4.9 % (q 0.980)

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -2.7 pts, CI [-4.6, -0.9] |
| A3b | fail | family faceswap-stills, threat faceswap |
| A4 | pass | pooled reals: Δ FPR -0.1 pts, CI [-0.6, +0.5] (blocks with a lower bound above +1.0, D-53); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 87.6 % | 85.0 % | -2.6 |
| F2 | 70 | 45.7 % | 47.1 % | +1.4 |
| F3 | 57 | 29.8 % | 21.1 % | -8.8 |
| F4 | 118 | 57.6 % | 53.4 % | -4.2 |
| F5 | 80 | 71.2 % | 71.2 % | +0.0 |
| F6 | 106 | 44.3 % | 44.3 % | +0.0 |
| F7 | 106 | 72.6 % | 67.9 % | -4.7 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 58.4 %, with 55.7 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 3.8 % | 5.0 % | +1.2 | [+0.0, +2.3] |
| R2 | 612 | 5.1 % | 4.1 % | -1.0 | [-1.8, +0.0] |
| R3 | 589 | 6.5 % | 6.1 % | -0.3 | [-1.3, +0.5] |
| R4 | 493 | 5.9 % | 6.1 % | +0.2 | [-0.6, +1.0] |
| R5 | 548 | 3.0 % | 2.9 % | -0.1 | [-1.5, +1.3] |
| R6 | 28 | 17.9 % | 17.9 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 67.1 % | 64.6 % | -2.5 | [-5.0, -0.6] |
| threat faceswap | 272 | 57.7 % | 71.3 % | +13.6 | [+8.1, +20.4] |
| threat gan_face | 153 | 39.9 % | 48.4 % | +8.5 | [+0.7, +15.7] |
| threat personalised | 115 | 96.5 % | 94.8 % | -1.7 | [-4.3, +0.0] |
| threat t2i | 1600 | 67.2 % | 65.4 % | -1.8 | [-2.6, -1.1] |
| family faceswap-stills | 272 | 57.7 % | 71.3 % | +13.6 | [+8.1, +20.4] |
| family flux1 | 85 | 67.1 % | 63.5 % | -3.5 | [-8.2, +0.0] |
| family flux2 | 70 | 42.9 % | 42.9 % | +0.0 | [-4.3, +4.2] |
| family gan-faces | 153 | 39.9 % | 48.4 % | +8.5 | [+0.7, +15.7] |
| family gpt-4o-image-1 | 41 | 41.5 % | 39.0 % | -2.4 | [-7.3, +0.0] |
| family gpt-image-1.5-2 | 57 | 19.3 % | 17.5 % | -1.8 | [-5.3, +0.0] |
| family hidream | 50 | 86.0 % | 84.0 % | -2.0 | [-6.4, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 78.7 % | 76.6 % | -2.1 | [-6.7, +0.0] |
| family imagen3 | 53 | 56.6 % | 49.1 % | -7.5 | [-15.1, -1.9] |
| family imagen4 | 106 | 64.2 % | 63.2 % | -0.9 | [-4.0, +2.0] |
| family legacy | 145 | 91.0 % | 91.0 % | +0.0 | [-2.1, +2.0] |
| family long-tail | 186 | 71.0 % | 71.5 % | +0.5 | [-1.7, +2.9] |
| family midjourney | 153 | 81.7 % | 79.7 % | -2.0 | [-4.6, +0.0] |
| family modern-t2i-faces | 115 | 67.8 % | 64.3 % | -3.5 | [-6.7, -0.8] |
| family nano-banana | 118 | 45.8 % | 41.5 % | -4.2 | [-8.5, -0.8] |
| family partial-edits | 161 | 67.1 % | 64.6 % | -2.5 | [-5.0, -0.6] |
| family personalised | 115 | 96.5 % | 94.8 % | -1.7 | [-4.3, +0.0] |
| family qwen-image | 106 | 37.7 % | 34.0 % | -3.8 | [-7.7, -0.9] |
| family recraft | 54 | 87.0 % | 87.0 % | +0.0 | [+0.0, +0.0] |
| family sd3 | 75 | 98.7 % | 98.7 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 80.0 % | 80.0 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 61.3 % | 58.8 % | -2.5 | [-6.3, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
