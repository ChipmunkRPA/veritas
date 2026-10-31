# Ablation: img.forensic.copymove

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.hf.ateeqq,img.forensic.copymove GATES='img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.forensic.copymove ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.hf.ateeqq
- Items (fit ∪ cal): 4993 (2692 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro +0.0 pts, CI [+0.0, +0.0] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR -0.1 pts, CI [-0.3, +0.2] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 37.3 % | 37.3 % | +0.0 |
| F2 | 70 | 42.9 % | 42.9 % | +0.0 |
| F3 | 57 | 40.4 % | 40.4 % | +0.0 |
| F4 | 118 | 49.2 % | 49.2 % | +0.0 |
| F5 | 80 | 40.0 % | 40.0 % | +0.0 |
| F6 | 106 | 34.9 % | 34.9 % | +0.0 |
| F7 | 106 | 32.1 % | 32.1 % | +0.0 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 39.5 %, with 39.5 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 4.5 % | 5.0 % | +0.5 | [+0.0, +1.2] |
| R2 | 611 | 4.4 % | 4.3 % | -0.2 | [-0.7, +0.0] |
| R3 | 588 | 10.7 % | 10.5 % | -0.2 | [-0.7, +0.3] |
| R4 | 493 | 4.1 % | 3.9 % | -0.2 | [-0.6, +0.0] |
| R5 | 548 | 1.5 % | 1.3 % | -0.2 | [-0.5, +0.0] |
| R6 | 28 | 7.1 % | 7.1 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 18.6 % | 18.6 % | +0.0 | [+0.0, +0.0] |
| threat faceswap | 272 | 0.4 % | 0.4 % | +0.0 | [+0.0, +0.0] |
| threat gan_face | 153 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| threat personalised | 115 | 9.6 % | 9.6 % | +0.0 | [+0.0, +0.0] |
| threat t2i | 1600 | 37.6 % | 37.4 % | -0.2 | [-0.6, +0.0] |
| family faceswap-stills | 272 | 0.4 % | 0.4 % | +0.0 | [+0.0, +0.0] |
| family flux1 | 85 | 61.2 % | 61.2 % | +0.0 | [+0.0, +0.0] |
| family flux2 | 70 | 42.9 % | 42.9 % | +0.0 | [+0.0, +0.0] |
| family gan-faces | 153 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| family gpt-4o-image-1 | 41 | 41.5 % | 41.5 % | +0.0 | [+0.0, +0.0] |
| family gpt-image-1.5-2 | 57 | 40.4 % | 40.4 % | +0.0 | [+0.0, +0.0] |
| family hidream | 50 | 62.0 % | 62.0 % | +0.0 | [+0.0, +0.0] |
| family hunyuanimage | 24 | 62.5 % | 62.5 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 29.8 % | 29.8 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 53 | 22.6 % | 22.6 % | +0.0 | [+0.0, +0.0] |
| family imagen4 | 106 | 32.1 % | 32.1 % | +0.0 | [+0.0, +0.0] |
| family legacy | 145 | 39.3 % | 37.9 % | -1.4 | [-3.7, +0.0] |
| family long-tail | 186 | 41.4 % | 40.3 % | -1.1 | [-3.3, +0.0] |
| family midjourney | 153 | 38.6 % | 38.6 % | +0.0 | [+0.0, +0.0] |
| family modern-t2i-faces | 115 | 2.6 % | 2.6 % | +0.0 | [+0.0, +0.0] |
| family nano-banana | 118 | 49.2 % | 49.2 % | +0.0 | [+0.0, +0.0] |
| family partial-edits | 161 | 18.6 % | 18.6 % | +0.0 | [+0.0, +0.0] |
| family personalised | 115 | 9.6 % | 9.6 % | +0.0 | [+0.0, +0.0] |
| family qwen-image | 106 | 34.9 % | 34.9 % | +0.0 | [+0.0, +0.0] |
| family recraft | 54 | 29.6 % | 29.6 % | +0.0 | [+0.0, +0.0] |
| family sd3 | 75 | 29.3 % | 29.3 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 31.4 % | 31.4 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 42.5 % | 42.5 % | +0.0 | [+0.0, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
