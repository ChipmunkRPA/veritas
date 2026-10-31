# Ablation: img.hf.ateeqq

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7,img.hf.ateeqq,img.hf.dima806 GATES='img.dfdc_b7=has_face;img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.ateeqq ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda, img.dfdc_b7, img.hf.dima806
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3
- Operating point (D-53): reals scoring exactly τ count as flagged with weight q (the rank rule's share); served rule (s ≥ τ) weighted FPR: without 5.0 % (q 0.230), with 5.0 % (q 0.365)

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +6.7 pts, CI [+4.6, +8.9] |
| A3b | pass | family gpt-4o-image-1, family imagen4, family qwen-image |
| A4 | pass | pooled reals: Δ FPR +0.0 pts, CI [-0.8, +0.9] (blocks with a lower bound above +1.0, D-53); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R1; above +2.0 on fewer than 150 reals (no term): R6 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 84.3 % | 87.6 % | +3.3 |
| F2 | 70 | 32.9 % | 38.6 % | +5.7 |
| F3 | 57 | 15.8 % | 19.3 % | +3.5 |
| F4 | 118 | 37.3 % | 46.6 % | +9.3 |
| F5 | 80 | 78.8 % | 81.2 % | +2.5 |
| F6 | 106 | 34.0 % | 44.3 % | +10.4 |
| F7 | 106 | 53.8 % | 66.0 % | +12.3 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 48.1 %, with 54.8 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.4 % | 4.5 % | +2.1 | [-1.3, +3.5] |
| R2 | 612 | 4.9 % | 5.2 % | +0.3 | [-0.9, +2.7] |
| R3 | 589 | 4.6 % | 5.8 % | +1.2 | [+0.0, +2.4] |
| R4 | 493 | 11.6 % | 7.2 % | -4.4 | [-6.8, -2.0] |
| R5 | 548 | 1.3 % | 1.6 % | +0.4 | [+0.0, +0.9] |
| R6 | 28 | 3.6 % | 14.3 % | +10.7 | [+0.0, +25.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 64.6 % | 65.8 % | +1.2 | [-1.2, +3.7] |
| threat faceswap | 272 | 67.6 % | 68.4 % | +0.7 | [-0.8, +2.2] |
| threat gan_face | 153 | 60.1 % | 62.1 % | +2.0 | [-0.7, +5.2] |
| threat personalised | 115 | 96.5 % | 97.4 % | +0.9 | [+0.0, +2.6] |
| threat t2i | 1600 | 61.4 % | 67.8 % | +6.4 | [+5.0, +7.9] |
| family faceswap-stills | 272 | 67.6 % | 68.4 % | +0.7 | [-0.8, +2.2] |
| family flux1 | 85 | 63.5 % | 69.4 % | +5.9 | [+1.2, +11.6] |
| family flux2 | 70 | 28.6 % | 35.7 % | +7.1 | [+1.5, +13.3] |
| family gan-faces | 153 | 60.1 % | 62.1 % | +2.0 | [-0.7, +5.2] |
| family gpt-4o-image-1 | 41 | 9.8 % | 26.8 % | +17.1 | [+4.9, +31.7] |
| family gpt-image-1.5-2 | 57 | 14.0 % | 15.8 % | +1.8 | [-3.5, +8.8] |
| family hidream | 50 | 80.0 % | 90.0 % | +10.0 | [+0.0, +20.8] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 74.5 % | 76.6 % | +2.1 | [+0.0, +6.8] |
| family imagen3 | 53 | 49.1 % | 54.7 % | +5.7 | [-5.7, +17.0] |
| family imagen4 | 106 | 49.1 % | 62.3 % | +13.2 | [+6.6, +20.0] |
| family legacy | 145 | 90.3 % | 95.9 % | +5.5 | [+0.7, +10.9] |
| family long-tail | 186 | 62.9 % | 71.0 % | +8.1 | [+4.7, +12.3] |
| family midjourney | 153 | 81.7 % | 85.0 % | +3.3 | [+0.0, +7.3] |
| family modern-t2i-faces | 115 | 58.3 % | 64.3 % | +6.1 | [+2.3, +11.3] |
| family nano-banana | 118 | 32.2 % | 41.5 % | +9.3 | [+4.2, +14.5] |
| family partial-edits | 161 | 64.6 % | 65.8 % | +1.2 | [-1.2, +3.7] |
| family personalised | 115 | 96.5 % | 97.4 % | +0.9 | [+0.0, +2.6] |
| family qwen-image | 106 | 29.2 % | 39.6 % | +10.4 | [+4.7, +17.0] |
| family recraft | 54 | 88.9 % | 92.6 % | +3.7 | [-3.6, +11.1] |
| family sd3 | 75 | 96.0 % | 98.7 % | +2.7 | [+0.0, +6.4] |
| family seedream3 | 35 | 82.9 % | 82.9 % | +0.0 | [-8.6, +8.6] |
| family seedream4 | 80 | 76.2 % | 77.5 % | +1.3 | [-3.8, +7.3] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
