# Ablation: img.hf.ateeqq

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=face.gend_clip_l14,img.commfor384,img.dda,img.dfdc_b7,img.hf.ateeqq,img.hf.dima806 GATES='face.gend_clip_l14=has_face;img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.ateeqq ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: face.gend_clip_l14, img.commfor384, img.dda, img.dfdc_b7, img.hf.dima806
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3
- Operating point (D-53): reals scoring exactly τ count as flagged with weight q (the rank rule's share); served rule (s ≥ τ) weighted FPR: without 4.9 % (q 0.762), with 5.0 % (q 0.413)

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +2.7 pts, CI [+0.7, +4.7] |
| A3b | pass | family flux2, family imagen4, family nano-banana |
| A4 | pass | pooled reals: Δ FPR +0.0 pts, CI [-0.7, +0.7] (blocks with a lower bound above +1.0, D-53); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R5; above +2.0 on fewer than 150 reals (no term): R6 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 84.3 % | 81.0 % | -3.3 |
| F2 | 70 | 32.9 % | 38.6 % | +5.7 |
| F3 | 57 | 14.0 % | 14.0 % | +0.0 |
| F4 | 118 | 37.3 % | 40.7 % | +3.4 |
| F5 | 80 | 81.2 % | 76.2 % | -5.0 |
| F6 | 106 | 33.0 % | 40.6 % | +7.5 |
| F7 | 106 | 52.8 % | 63.2 % | +10.4 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 47.9 %, with 50.6 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 6.4 % | 7.3 % | +0.9 | [-1.0, +2.0] |
| R2 | 612 | 4.2 % | 3.4 % | -0.8 | [-1.6, +0.4] |
| R3 | 589 | 4.4 % | 4.4 % | +0.0 | [-1.2, +1.2] |
| R4 | 493 | 7.9 % | 4.1 % | -3.9 | [-6.1, -1.6] |
| R5 | 548 | 1.8 % | 5.2 % | +3.4 | [+1.8, +5.1] |
| R6 | 28 | 3.6 % | 10.7 % | +7.1 | [+0.0, +19.2] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 64.0 % | 65.2 % | +1.2 | [-1.2, +3.7] |
| threat faceswap | 272 | 77.6 % | 79.8 % | +2.2 | [-0.8, +5.6] |
| threat gan_face | 153 | 68.0 % | 59.5 % | -8.5 | [-14.4, -3.3] |
| threat personalised | 115 | 97.4 % | 93.9 % | -3.5 | [-7.0, -0.9] |
| threat t2i | 1600 | 60.6 % | 65.6 % | +5.0 | [+3.6, +6.4] |
| family faceswap-stills | 272 | 77.6 % | 79.8 % | +2.2 | [-0.8, +5.6] |
| family flux1 | 85 | 62.4 % | 64.7 % | +2.4 | [-2.3, +7.1] |
| family flux2 | 70 | 27.1 % | 37.1 % | +10.0 | [+4.1, +17.4] |
| family gan-faces | 153 | 68.0 % | 59.5 % | -8.5 | [-14.4, -3.3] |
| family gpt-4o-image-1 | 41 | 9.8 % | 17.1 % | +7.3 | [-2.4, +19.5] |
| family gpt-image-1.5-2 | 57 | 14.0 % | 12.3 % | -1.8 | [-5.3, +0.0] |
| family hidream | 50 | 78.0 % | 86.0 % | +8.0 | [+0.0, +18.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 74.5 % | 72.3 % | -2.1 | [-10.0, +4.4] |
| family imagen3 | 53 | 43.4 % | 50.9 % | +7.5 | [-1.9, +17.0] |
| family imagen4 | 106 | 49.1 % | 61.3 % | +12.3 | [+5.7, +19.2] |
| family legacy | 145 | 86.9 % | 94.5 % | +7.6 | [+2.8, +12.7] |
| family long-tail | 186 | 63.4 % | 70.4 % | +7.0 | [+3.0, +11.1] |
| family midjourney | 153 | 81.7 % | 81.7 % | +0.0 | [-3.8, +3.9] |
| family modern-t2i-faces | 115 | 59.1 % | 64.3 % | +5.2 | [+1.5, +10.0] |
| family nano-banana | 118 | 30.5 % | 40.7 % | +10.2 | [+5.0, +16.0] |
| family partial-edits | 161 | 64.0 % | 65.2 % | +1.2 | [-1.2, +3.7] |
| family personalised | 115 | 97.4 % | 93.9 % | -3.5 | [-7.0, -0.9] |
| family qwen-image | 106 | 29.2 % | 37.7 % | +8.5 | [+2.9, +14.8] |
| family recraft | 54 | 87.0 % | 81.5 % | -5.6 | [-13.5, +1.9] |
| family sd3 | 75 | 96.0 % | 98.7 % | +2.7 | [+0.0, +6.4] |
| family seedream3 | 35 | 82.9 % | 82.9 % | +0.0 | [-8.6, +8.6] |
| family seedream4 | 80 | 75.0 % | 73.8 % | -1.2 | [-5.1, +2.5] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
