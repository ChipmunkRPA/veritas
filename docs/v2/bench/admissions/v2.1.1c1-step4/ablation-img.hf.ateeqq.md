# Ablation: img.hf.ateeqq

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.dfdc_b7,img.hf.ateeqq,img.hf.dima806 GATES='img.dfdc_b7=has_face;img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.ateeqq ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.dfdc_b7, img.hf.dima806
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3
- Operating point (D-53): reals scoring exactly τ count as flagged with weight q (the rank rule's share); served rule (s ≥ τ) weighted FPR: without 4.9 % (q 0.897), with 5.0 % (q 0.407)

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +11.3 pts, CI [+8.6, +14.1] |
| A3b | pass | family flux2, family imagen4, family qwen-image |
| A4 | pass | pooled reals: Δ FPR -0.1 pts, CI [-1.1, +1.1] (blocks with a lower bound above +1.0, D-53); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): R6 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 83.0 % | 87.6 % | +4.6 |
| F2 | 70 | 32.9 % | 45.7 % | +12.9 |
| F3 | 57 | 15.8 % | 29.8 % | +14.0 |
| F4 | 118 | 44.9 % | 57.6 % | +12.7 |
| F5 | 80 | 63.7 % | 71.2 % | +7.5 |
| F6 | 106 | 31.1 % | 44.3 % | +13.2 |
| F7 | 106 | 58.5 % | 72.6 % | +14.2 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 47.1 %, with 58.4 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.1 % | 3.8 % | +1.7 | [+0.0, +2.5] |
| R2 | 612 | 7.8 % | 5.1 % | -2.8 | [-5.0, +0.6] |
| R3 | 589 | 4.8 % | 6.5 % | +1.7 | [+0.2, +3.2] |
| R4 | 493 | 7.7 % | 5.9 % | -1.8 | [-4.2, +0.6] |
| R5 | 548 | 2.0 % | 3.0 % | +1.0 | [+0.3, +1.9] |
| R6 | 28 | 10.7 % | 17.9 % | +7.1 | [-6.7, +21.4] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 66.5 % | 67.1 % | +0.6 | [-1.9, +3.7] |
| threat faceswap | 272 | 59.6 % | 57.7 % | -1.8 | [-4.4, +0.3] |
| threat gan_face | 153 | 34.6 % | 39.9 % | +5.2 | [+1.3, +9.8] |
| threat personalised | 115 | 95.7 % | 96.5 % | +0.9 | [+0.0, +2.6] |
| threat t2i | 1600 | 61.5 % | 67.2 % | +5.8 | [+4.2, +7.3] |
| family faceswap-stills | 272 | 59.6 % | 57.7 % | -1.8 | [-4.4, +0.3] |
| family flux1 | 85 | 61.2 % | 67.1 % | +5.9 | [+1.2, +11.8] |
| family flux2 | 70 | 28.6 % | 42.9 % | +14.3 | [+4.5, +24.7] |
| family gan-faces | 153 | 34.6 % | 39.9 % | +5.2 | [+1.3, +9.8] |
| family gpt-4o-image-1 | 41 | 34.1 % | 41.5 % | +7.3 | [+0.0, +17.1] |
| family gpt-image-1.5-2 | 57 | 14.0 % | 19.3 % | +5.3 | [+0.0, +12.3] |
| family hidream | 50 | 80.0 % | 86.0 % | +6.0 | [-2.1, +14.8] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 72.3 % | 78.7 % | +6.4 | [+0.0, +13.6] |
| family imagen3 | 53 | 52.8 % | 56.6 % | +3.8 | [-5.7, +13.2] |
| family imagen4 | 106 | 53.8 % | 64.2 % | +10.4 | [+4.8, +16.0] |
| family legacy | 145 | 82.1 % | 91.0 % | +9.0 | [+4.1, +14.4] |
| family long-tail | 186 | 63.4 % | 71.0 % | +7.5 | [+2.9, +12.4] |
| family midjourney | 153 | 80.4 % | 81.7 % | +1.3 | [-2.6, +5.4] |
| family modern-t2i-faces | 115 | 66.1 % | 67.8 % | +1.7 | [+0.0, +4.6] |
| family nano-banana | 118 | 42.4 % | 45.8 % | +3.4 | [-4.2, +11.0] |
| family partial-edits | 161 | 66.5 % | 67.1 % | +0.6 | [-1.9, +3.7] |
| family personalised | 115 | 95.7 % | 96.5 % | +0.9 | [+0.0, +2.6] |
| family qwen-image | 106 | 26.4 % | 37.7 % | +11.3 | [+4.8, +18.5] |
| family recraft | 54 | 83.3 % | 87.0 % | +3.7 | [-5.4, +13.0] |
| family sd3 | 75 | 96.0 % | 98.7 % | +2.7 | [+0.0, +6.4] |
| family seedream3 | 35 | 80.0 % | 80.0 % | +0.0 | [-8.6, +8.6] |
| family seedream4 | 80 | 60.0 % | 61.3 % | +1.3 | [-3.8, +6.4] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
