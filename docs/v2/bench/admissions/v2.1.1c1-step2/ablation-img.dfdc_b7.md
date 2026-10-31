# Ablation: img.dfdc_b7

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7,img.hf.ateeqq,img.hf.dima806 GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.dfdc_b7 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda, img.hf.ateeqq, img.hf.dima806
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3
- Operating point (D-53): reals scoring exactly τ count as flagged with weight q (the rank rule's share); served rule (s ≥ τ) weighted FPR: without 5.0 % (q 0.537), with 5.0 % (q 0.317)

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +6.3 pts, CI [+4.4, +8.3] |
| A3b | pass | family faceswap-stills, family imagen3, threat faceswap |
| A4 | pass | pooled reals: Δ FPR -0.0 pts, CI [-0.9, +0.7] (blocks with a lower bound above +1.0, D-53); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R1; above +2.0 on fewer than 150 reals (no term): R6 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 78.4 % | 81.7 % | +3.3 |
| F2 | 70 | 31.4 % | 38.6 % | +7.1 |
| F3 | 57 | 10.5 % | 14.0 % | +3.5 |
| F4 | 118 | 39.0 % | 40.7 % | +1.7 |
| F5 | 80 | 72.5 % | 77.5 % | +5.0 |
| F6 | 106 | 33.0 % | 45.3 % | +12.3 |
| F7 | 106 | 53.8 % | 65.1 % | +11.3 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 45.5 %, with 51.8 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 1.4 % | 3.8 % | +2.4 | [+0.0, +3.9] |
| R2 | 612 | 2.5 % | 3.6 % | +1.1 | [+0.3, +1.7] |
| R3 | 589 | 3.7 % | 5.4 % | +1.7 | [+0.7, +2.8] |
| R4 | 493 | 3.9 % | 4.5 % | +0.7 | [-0.1, +1.6] |
| R5 | 548 | 13.2 % | 7.1 % | -6.1 | [-8.7, -3.7] |
| R6 | 28 | 3.6 % | 10.7 % | +7.1 | [+0.0, +19.2] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 63.4 % | 65.8 % | +2.5 | [+0.6, +5.0] |
| threat faceswap | 272 | 34.9 % | 63.6 % | +28.7 | [+17.4, +39.9] |
| threat gan_face | 153 | 49.0 % | 51.6 % | +2.6 | [-2.6, +7.9] |
| threat personalised | 115 | 94.8 % | 94.8 % | +0.0 | [-2.6, +2.6] |
| threat t2i | 1600 | 63.6 % | 66.9 % | +3.4 | [+2.4, +4.4] |
| family faceswap-stills | 272 | 34.9 % | 63.6 % | +28.7 | [+17.4, +39.9] |
| family flux1 | 85 | 62.4 % | 67.1 % | +4.7 | [+1.1, +9.8] |
| family flux2 | 70 | 32.9 % | 38.6 % | +5.7 | [+1.4, +11.6] |
| family gan-faces | 153 | 49.0 % | 51.6 % | +2.6 | [-2.6, +7.9] |
| family gpt-4o-image-1 | 41 | 12.2 % | 22.0 % | +9.8 | [+2.4, +19.5] |
| family gpt-image-1.5-2 | 57 | 10.5 % | 14.0 % | +3.5 | [+0.0, +8.8] |
| family hidream | 50 | 84.0 % | 88.0 % | +4.0 | [+0.0, +10.2] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 72.3 % | 72.3 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 53 | 43.4 % | 56.6 % | +13.2 | [+3.8, +22.6] |
| family imagen4 | 106 | 57.5 % | 61.3 % | +3.8 | [+0.0, +8.5] |
| family legacy | 145 | 93.8 % | 94.5 % | +0.7 | [+0.0, +2.2] |
| family long-tail | 186 | 66.7 % | 70.4 % | +3.8 | [+1.1, +7.0] |
| family midjourney | 153 | 79.7 % | 82.4 % | +2.6 | [+0.6, +5.5] |
| family modern-t2i-faces | 115 | 68.7 % | 67.8 % | -0.9 | [-4.7, +3.1] |
| family nano-banana | 118 | 39.0 % | 40.7 % | +1.7 | [+0.0, +4.2] |
| family partial-edits | 161 | 63.4 % | 65.8 % | +2.5 | [+0.6, +5.0] |
| family personalised | 115 | 94.8 % | 94.8 % | +0.0 | [-2.6, +2.6] |
| family qwen-image | 106 | 32.1 % | 40.6 % | +8.5 | [+3.7, +14.7] |
| family recraft | 54 | 81.5 % | 83.3 % | +1.9 | [+0.0, +5.8] |
| family sd3 | 75 | 97.3 % | 98.7 % | +1.3 | [+0.0, +4.1] |
| family seedream3 | 35 | 82.9 % | 82.9 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 73.8 % | 77.5 % | +3.7 | [+0.0, +8.8] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
