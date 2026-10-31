# Ablation: img.dfdc_b7

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.dfdc_b7,img.hf.ateeqq,img.hf.dima806 GATES='img.dfdc_b7=has_face;img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.dfdc_b7 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.hf.ateeqq, img.hf.dima806
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3
- Operating point (D-53): reals scoring exactly τ count as flagged with weight q (the rank rule's share); served rule (s ≥ τ) weighted FPR: without 6.7 % (q 0.166), with 5.0 % (q 0.407)

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +7.6 pts, CI [+5.5, +10.0] |
| A3b | pass | family faceswap-stills, threat faceswap |
| A4 | pass | pooled reals: Δ FPR +0.0 pts, CI [-0.5, +0.5] (blocks with a lower bound above +1.0, D-53); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 82.4 % | 87.6 % | +5.2 |
| F2 | 70 | 41.4 % | 45.7 % | +4.3 |
| F3 | 57 | 15.8 % | 29.8 % | +14.0 |
| F4 | 118 | 47.5 % | 57.6 % | +10.2 |
| F5 | 80 | 63.7 % | 71.2 % | +7.5 |
| F6 | 106 | 37.7 % | 44.3 % | +6.6 |
| F7 | 106 | 67.0 % | 72.6 % | +5.7 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 50.8 %, with 58.4 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 3.3 % | 3.8 % | +0.5 | [+0.0, +1.0] |
| R2 | 612 | 4.1 % | 5.1 % | +1.0 | [+0.0, +1.8] |
| R3 | 589 | 5.4 % | 6.5 % | +1.0 | [+0.3, +1.9] |
| R4 | 493 | 6.0 % | 5.9 % | -0.1 | [-0.3, -0.0] |
| R5 | 548 | 5.3 % | 3.0 % | -2.3 | [-3.7, -0.8] |
| R6 | 28 | 17.9 % | 17.9 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 65.3 % | 67.1 % | +1.8 | [+0.0, +4.0] |
| threat faceswap | 272 | 15.2 % | 57.7 % | +42.5 | [+29.7, +53.4] |
| threat gan_face | 153 | 35.2 % | 39.9 % | +4.7 | [-1.0, +10.7] |
| threat personalised | 115 | 96.5 % | 96.5 % | +0.0 | [+0.0, +0.0] |
| threat t2i | 1600 | 66.5 % | 67.2 % | +0.7 | [+0.1, +1.4] |
| family faceswap-stills | 272 | 15.2 % | 57.7 % | +42.5 | [+29.7, +53.4] |
| family flux1 | 85 | 64.7 % | 67.1 % | +2.4 | [+0.0, +6.0] |
| family flux2 | 70 | 42.9 % | 42.9 % | +0.0 | [+0.0, +0.0] |
| family gan-faces | 153 | 35.2 % | 39.9 % | +4.7 | [-1.0, +10.7] |
| family gpt-4o-image-1 | 41 | 39.0 % | 41.5 % | +2.4 | [+0.0, +7.3] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 19.3 % | +3.5 | [+0.0, +8.8] |
| family hidream | 50 | 86.0 % | 86.0 % | +0.0 | [+0.0, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 78.7 % | 78.7 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 53 | 54.7 % | 56.6 % | +1.9 | [+0.0, +5.7] |
| family imagen4 | 106 | 66.0 % | 64.2 % | -1.9 | [-4.9, +0.0] |
| family legacy | 145 | 90.3 % | 91.0 % | +0.7 | [+0.0, +2.2] |
| family long-tail | 186 | 69.9 % | 71.0 % | +1.1 | [+0.0, +2.7] |
| family midjourney | 153 | 81.0 % | 81.7 % | +0.7 | [-1.3, +2.9] |
| family modern-t2i-faces | 115 | 69.6 % | 67.8 % | -1.7 | [-4.2, +0.0] |
| family nano-banana | 118 | 43.2 % | 45.8 % | +2.5 | [+0.0, +5.9] |
| family partial-edits | 161 | 65.3 % | 67.1 % | +1.8 | [+0.0, +4.0] |
| family personalised | 115 | 96.5 % | 96.5 % | +0.0 | [+0.0, +0.0] |
| family qwen-image | 106 | 35.8 % | 37.7 % | +1.9 | [-1.9, +5.8] |
| family recraft | 54 | 85.2 % | 87.0 % | +1.9 | [-3.8, +8.6] |
| family sd3 | 75 | 98.7 % | 98.7 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 80.0 % | 80.0 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 61.3 % | 61.3 % | +0.0 | [+0.0, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
