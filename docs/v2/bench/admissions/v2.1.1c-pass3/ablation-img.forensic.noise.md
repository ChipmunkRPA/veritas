# Ablation: img.forensic.noise

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=face.gend_clip_l14,img.forensic.noise GATES='face.gend_clip_l14=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.forensic.noise ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: face.gend_clip_l14
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -1.0 pts, CI [-2.0, -0.2] |
| A3b | fail | family modern-t2i-faces |
| A4 | fail | pooled reals: Δ FPR +0.1 pts, CI [-1.1, +1.1] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 3.9 % | 4.6 % | +0.7 |
| F2 | 70 | 7.1 % | 4.3 % | -2.9 |
| F3 | 57 | 0.0 % | 0.0 % | +0.0 |
| F4 | 118 | 0.8 % | 0.0 % | -0.8 |
| F5 | 80 | 3.8 % | 2.5 % | -1.2 |
| F6 | 106 | 0.9 % | 0.9 % | +0.0 |
| F7 | 106 | 6.6 % | 3.8 % | -2.8 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 3.3 %, with 2.3 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 14.6 % | 12.0 % | -2.6 | [-6.9, +3.1] |
| R2 | 612 | 0.5 % | 0.2 % | -0.3 | [-0.9, +0.0] |
| R3 | 589 | 3.4 % | 4.6 % | +1.2 | [+0.2, +2.4] |
| R4 | 493 | 1.2 % | 1.8 % | +0.6 | [-0.4, +1.6] |
| R5 | 548 | 5.1 % | 6.2 % | +1.1 | [-0.2, +2.4] |
| R6 | 28 | 3.6 % | 3.6 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 11.2 % | 13.0 % | +1.9 | [+0.0, +4.3] |
| threat faceswap | 272 | 74.3 % | 77.6 % | +3.3 | [+0.9, +6.3] |
| threat gan_face | 153 | 47.7 % | 44.4 % | -3.3 | [-7.8, +0.7] |
| threat personalised | 115 | 33.0 % | 38.3 % | +5.2 | [+0.0, +10.4] |
| threat t2i | 1600 | 3.1 % | 4.9 % | +1.9 | [+1.0, +2.8] |
| family faceswap-stills | 272 | 74.3 % | 77.6 % | +3.3 | [+0.9, +6.3] |
| family flux1 | 85 | 1.2 % | 1.2 % | +0.0 | [+0.0, +0.0] |
| family flux2 | 70 | 10.0 % | 8.6 % | -1.4 | [-4.5, +0.0] |
| family gan-faces | 153 | 47.7 % | 44.4 % | -3.3 | [-7.8, +0.7] |
| family gpt-4o-image-1 | 41 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| family gpt-image-1.5-2 | 57 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| family hidream | 50 | 2.0 % | 6.0 % | +4.0 | [+0.0, +10.0] |
| family hunyuanimage | 24 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 0.0 % | 2.1 % | +2.1 | [+0.0, +6.0] |
| family imagen3 | 53 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| family imagen4 | 106 | 3.8 % | 4.7 % | +0.9 | [-2.0, +4.7] |
| family legacy | 145 | 2.8 % | 6.2 % | +3.4 | [+0.0, +7.1] |
| family long-tail | 186 | 4.8 % | 10.2 % | +5.4 | [+2.1, +8.8] |
| family midjourney | 153 | 3.3 % | 5.2 % | +2.0 | [+0.0, +4.6] |
| family modern-t2i-faces | 115 | 6.1 % | 16.5 % | +10.4 | [+4.2, +18.5] |
| family nano-banana | 118 | 1.7 % | 0.0 % | -1.7 | [-4.2, +0.0] |
| family partial-edits | 161 | 11.2 % | 13.0 % | +1.9 | [+0.0, +4.3] |
| family personalised | 115 | 33.0 % | 38.3 % | +5.2 | [+0.0, +10.4] |
| family qwen-image | 106 | 0.9 % | 0.9 % | +0.0 | [+0.0, +0.0] |
| family recraft | 54 | 3.7 % | 1.9 % | -1.9 | [-5.8, +0.0] |
| family sd3 | 75 | 2.7 % | 2.7 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 5.7 % | 5.7 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 2.5 % | 2.5 % | +0.0 | [+0.0, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
