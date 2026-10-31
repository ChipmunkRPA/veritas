# Ablation: face.gend_clip_l14

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=face.gend_clip_l14,img.commfor384,img.dda,img.dfdc_b7,img.hf.ateeqq,img.hf.dima806 GATES='face.gend_clip_l14=has_face;img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=face.gend_clip_l14 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda, img.dfdc_b7, img.hf.ateeqq, img.hf.dima806
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (Celeb-DF v3, DeepSpeak v1.1, DeepSpeak v2, FFIW, FaceForensics++)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3
- Operating point (D-53): reals scoring exactly τ count as flagged with weight q (the rank rule's share); served rule (s ≥ τ) weighted FPR: without 5.0 % (q 0.317), with 5.0 % (q 0.413)

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -1.2 pts, CI [-2.2, -0.4] |
| A3b | fail | family faceswap-stills, threat faceswap |
| A4 | pass | pooled reals: Δ FPR -0.2 pts, CI [-1.1, +1.1] (blocks with a lower bound above +1.0, D-53); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R1; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 81.7 % | 81.0 % | -0.7 |
| F2 | 70 | 38.6 % | 38.6 % | +0.0 |
| F3 | 57 | 14.0 % | 14.0 % | +0.0 |
| F4 | 118 | 40.7 % | 40.7 % | +0.0 |
| F5 | 80 | 77.5 % | 76.2 % | -1.3 |
| F6 | 106 | 45.3 % | 40.6 % | -4.7 |
| F7 | 106 | 65.1 % | 63.2 % | -1.9 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 51.8 %, with 50.6 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 3.8 % | 7.3 % | +3.5 | [-0.5, +7.6] |
| R2 | 612 | 3.6 % | 3.4 % | -0.2 | [-0.7, +0.0] |
| R3 | 589 | 5.4 % | 4.4 % | -1.0 | [-1.9, -0.3] |
| R4 | 493 | 4.5 % | 4.1 % | -0.5 | [-1.1, +0.0] |
| R5 | 548 | 7.1 % | 5.2 % | -1.9 | [-3.4, -0.5] |
| R6 | 28 | 10.7 % | 10.7 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 65.8 % | 65.2 % | -0.6 | [-1.9, +0.0] |
| threat faceswap | 272 | 63.6 % | 79.8 % | +16.2 | [+6.6, +26.3] |
| threat gan_face | 153 | 51.6 % | 59.5 % | +7.8 | [+2.6, +13.7] |
| threat personalised | 115 | 94.8 % | 93.9 % | -0.9 | [-2.6, +0.0] |
| threat t2i | 1600 | 66.9 % | 65.6 % | -1.4 | [-2.1, -0.8] |
| family faceswap-stills | 272 | 63.6 % | 79.8 % | +16.2 | [+6.6, +26.3] |
| family flux1 | 85 | 67.1 % | 64.7 % | -2.4 | [-6.1, +0.0] |
| family flux2 | 70 | 38.6 % | 37.1 % | -1.4 | [-4.4, +0.0] |
| family gan-faces | 153 | 51.6 % | 59.5 % | +7.8 | [+2.6, +13.7] |
| family gpt-4o-image-1 | 41 | 22.0 % | 17.1 % | -4.9 | [-12.2, +0.0] |
| family gpt-image-1.5-2 | 57 | 14.0 % | 12.3 % | -1.8 | [-5.3, +0.0] |
| family hidream | 50 | 88.0 % | 86.0 % | -2.0 | [-6.4, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 72.3 % | 72.3 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 53 | 56.6 % | 50.9 % | -5.7 | [-13.2, +0.0] |
| family imagen4 | 106 | 61.3 % | 61.3 % | +0.0 | [-2.8, +2.9] |
| family legacy | 145 | 94.5 % | 94.5 % | +0.0 | [+0.0, +0.0] |
| family long-tail | 186 | 70.4 % | 70.4 % | +0.0 | [-2.3, +2.2] |
| family midjourney | 153 | 82.4 % | 81.7 % | -0.7 | [-2.0, +0.0] |
| family modern-t2i-faces | 115 | 67.8 % | 64.3 % | -3.5 | [-6.6, -0.8] |
| family nano-banana | 118 | 40.7 % | 40.7 % | +0.0 | [+0.0, +0.0] |
| family partial-edits | 161 | 65.8 % | 65.2 % | -0.6 | [-1.9, +0.0] |
| family personalised | 115 | 94.8 % | 93.9 % | -0.9 | [-2.6, +0.0] |
| family qwen-image | 106 | 40.6 % | 37.7 % | -2.8 | [-7.6, +1.0] |
| family recraft | 54 | 83.3 % | 81.5 % | -1.9 | [-5.8, +0.0] |
| family sd3 | 75 | 98.7 % | 98.7 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 82.9 % | 82.9 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 77.5 % | 73.8 % | -3.7 | [-8.8, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
