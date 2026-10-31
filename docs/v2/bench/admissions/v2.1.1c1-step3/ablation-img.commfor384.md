# Ablation: img.commfor384

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7,img.hf.ateeqq,img.hf.dima806 GATES='img.dfdc_b7=has_face;img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.commfor384 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.dfdc_b7, img.hf.ateeqq, img.hf.dima806
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (OwensLab/CommunityForensics-Eval)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3
- Operating point (D-53): reals scoring exactly τ count as flagged with weight q (the rank rule's share); served rule (s ≥ τ) weighted FPR: without 5.0 % (q 0.407), with 5.0 % (q 0.365)

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -3.6 pts, CI [-6.5, -0.9] |
| A3b | fail | family faceswap-stills, family gan-faces, family seedream4, threat faceswap, threat gan_face |
| A4 | pass | pooled reals: Δ FPR -0.1 pts, CI [-0.8, +0.6] (blocks with a lower bound above +1.0, D-53); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 87.6 % | 87.6 % | +0.0 |
| F2 | 70 | 45.7 % | 38.6 % | -7.1 |
| F3 | 57 | 29.8 % | 19.3 % | -10.5 |
| F4 | 118 | 57.6 % | 46.6 % | -11.0 |
| F5 | 80 | 71.2 % | 81.2 % | +10.0 |
| F6 | 106 | 44.3 % | 44.3 % | +0.0 |
| F7 | 106 | 72.6 % | 66.0 % | -6.6 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 58.4 %, with 54.8 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 3.8 % | 4.5 % | +0.7 | [-1.4, +1.8] |
| R2 | 612 | 5.1 % | 5.2 % | +0.2 | [-1.3, +2.8] |
| R3 | 589 | 6.5 % | 5.8 % | -0.7 | [-2.4, +1.0] |
| R4 | 493 | 5.9 % | 7.2 % | +1.3 | [-0.2, +3.0] |
| R5 | 548 | 3.0 % | 1.6 % | -1.4 | [-2.4, -0.4] |
| R6 | 28 | 17.9 % | 14.3 % | -3.6 | [-12.5, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 67.1 % | 65.8 % | -1.2 | [-5.6, +3.1] |
| threat faceswap | 272 | 57.7 % | 68.4 % | +10.7 | [+6.7, +16.0] |
| threat gan_face | 153 | 39.9 % | 62.1 % | +22.2 | [+13.7, +30.1] |
| threat personalised | 115 | 96.5 % | 97.4 % | +0.9 | [-2.6, +4.3] |
| threat t2i | 1600 | 67.2 % | 67.8 % | +0.6 | [-1.0, +2.1] |
| family faceswap-stills | 272 | 57.7 % | 68.4 % | +10.7 | [+6.7, +16.0] |
| family flux1 | 85 | 67.1 % | 69.4 % | +2.4 | [-3.7, +8.4] |
| family flux2 | 70 | 42.9 % | 35.7 % | -7.1 | [-14.3, +0.0] |
| family gan-faces | 153 | 39.9 % | 62.1 % | +22.2 | [+13.7, +30.1] |
| family gpt-4o-image-1 | 41 | 41.5 % | 26.8 % | -14.6 | [-26.8, -4.9] |
| family gpt-image-1.5-2 | 57 | 19.3 % | 15.8 % | -3.5 | [-12.3, +7.0] |
| family hidream | 50 | 86.0 % | 90.0 % | +4.0 | [+0.0, +10.2] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 78.7 % | 76.6 % | -2.1 | [-11.4, +6.7] |
| family imagen3 | 53 | 56.6 % | 54.7 % | -1.9 | [-11.3, +7.5] |
| family imagen4 | 106 | 64.2 % | 62.3 % | -1.9 | [-8.9, +4.7] |
| family legacy | 145 | 91.0 % | 95.9 % | +4.8 | [+2.0, +8.5] |
| family long-tail | 186 | 71.0 % | 71.0 % | +0.0 | [-3.6, +3.7] |
| family midjourney | 153 | 81.7 % | 85.0 % | +3.3 | [+0.0, +6.7] |
| family modern-t2i-faces | 115 | 67.8 % | 64.3 % | -3.5 | [-9.5, +2.7] |
| family nano-banana | 118 | 45.8 % | 41.5 % | -4.2 | [-11.0, +2.6] |
| family partial-edits | 161 | 67.1 % | 65.8 % | -1.2 | [-5.6, +3.1] |
| family personalised | 115 | 96.5 % | 97.4 % | +0.9 | [-2.6, +4.3] |
| family qwen-image | 106 | 37.7 % | 39.6 % | +1.9 | [-2.8, +6.7] |
| family recraft | 54 | 87.0 % | 92.6 % | +5.6 | [+0.0, +12.5] |
| family sd3 | 75 | 98.7 % | 98.7 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 80.0 % | 82.9 % | +2.9 | [-5.9, +14.3] |
| family seedream4 | 80 | 61.3 % | 77.5 % | +16.2 | [+6.3, +26.2] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
