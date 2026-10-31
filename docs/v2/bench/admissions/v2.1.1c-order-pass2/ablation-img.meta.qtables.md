# Ablation: img.meta.qtables

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.dfdc_b7,img.meta.qtables GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.meta.qtables ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.dfdc_b7
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (vb-mini R1/R2 reals are the reference set; their false-positive rate is measured leave-one-out)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro +0.0 pts, CI [+0.0, +0.0] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR +0.0 pts, CI [+0.0, +0.0] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 81.7 % | 81.7 % | +0.0 |
| F2 | 70 | 30.0 % | 30.0 % | +0.0 |
| F3 | 57 | 15.8 % | 15.8 % | +0.0 |
| F4 | 118 | 44.9 % | 44.9 % | +0.0 |
| F5 | 80 | 61.3 % | 61.3 % | +0.0 |
| F6 | 106 | 27.4 % | 27.4 % | +0.0 |
| F7 | 106 | 60.4 % | 60.4 % | +0.0 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 45.9 %, with 45.9 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.8 % | 2.8 % | +0.0 | [+0.0, +0.0] |
| R2 | 612 | 8.8 % | 8.8 % | +0.0 | [+0.0, +0.0] |
| R3 | 589 | 4.9 % | 4.9 % | +0.0 | [+0.0, +0.0] |
| R4 | 493 | 8.7 % | 8.7 % | +0.0 | [+0.0, +0.0] |
| R5 | 548 | 1.5 % | 1.5 % | +0.0 | [+0.0, +0.0] |
| R6 | 28 | 14.3 % | 14.3 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 66.5 % | 66.5 % | +0.0 | [+0.0, +0.0] |
| threat faceswap | 272 | 55.5 % | 55.5 % | +0.0 | [+0.0, +0.0] |
| threat gan_face | 153 | 25.5 % | 25.5 % | +0.0 | [+0.0, +0.0] |
| threat personalised | 115 | 95.7 % | 95.7 % | +0.0 | [+0.0, +0.0] |
| threat t2i | 1600 | 63.8 % | 63.8 % | +0.0 | [+0.0, +0.0] |
| family faceswap-stills | 272 | 55.5 % | 55.5 % | +0.0 | [+0.0, +0.0] |
| family flux1 | 85 | 61.2 % | 61.2 % | +0.0 | [+0.0, +0.0] |
| family flux2 | 70 | 30.0 % | 30.0 % | +0.0 | [+0.0, +0.0] |
| family gan-faces | 153 | 25.5 % | 25.5 % | +0.0 | [+0.0, +0.0] |
| family gpt-4o-image-1 | 41 | 34.1 % | 34.1 % | +0.0 | [+0.0, +0.0] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 15.8 % | +0.0 | [+0.0, +0.0] |
| family hidream | 50 | 82.0 % | 82.0 % | +0.0 | [+0.0, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 76.6 % | 76.6 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 53 | 52.8 % | 52.8 % | +0.0 | [+0.0, +0.0] |
| family imagen4 | 106 | 59.4 % | 59.4 % | +0.0 | [+0.0, +0.0] |
| family legacy | 145 | 89.0 % | 89.0 % | +0.0 | [+0.0, +0.0] |
| family long-tail | 186 | 65.6 % | 65.6 % | +0.0 | [+0.0, +0.0] |
| family midjourney | 153 | 81.7 % | 81.7 % | +0.0 | [+0.0, +0.0] |
| family modern-t2i-faces | 115 | 66.1 % | 66.1 % | +0.0 | [+0.0, +0.0] |
| family nano-banana | 118 | 44.9 % | 44.9 % | +0.0 | [+0.0, +0.0] |
| family partial-edits | 161 | 66.5 % | 66.5 % | +0.0 | [+0.0, +0.0] |
| family personalised | 115 | 95.7 % | 95.7 % | +0.0 | [+0.0, +0.0] |
| family qwen-image | 106 | 28.3 % | 28.3 % | +0.0 | [+0.0, +0.0] |
| family recraft | 54 | 87.0 % | 87.0 % | +0.0 | [+0.0, +0.0] |
| family sd3 | 75 | 97.3 % | 97.3 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 82.9 % | 82.9 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 61.3 % | 61.3 % | +0.0 | [+0.0, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
