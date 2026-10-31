# Ablation: pe.jpeg_dq.support

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=face.gend_clip_l14,pe.jpeg_dq.support GATES='face.gend_clip_l14=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=pe.jpeg_dq.support ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: face.gend_clip_l14
- Items (fit ∪ cal): 5001 (2697 real, 2304 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro +0.0 pts, CI [+0.0, +0.0] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR +0.0 pts, CI [+0.0, +0.0] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 3.9 % | 3.9 % | +0.0 |
| F2 | 70 | 7.1 % | 7.1 % | +0.0 |
| F3 | 57 | 0.0 % | 0.0 % | +0.0 |
| F4 | 118 | 0.8 % | 0.8 % | +0.0 |
| F5 | 80 | 3.8 % | 3.8 % | +0.0 |
| F6 | 106 | 0.9 % | 0.9 % | +0.0 |
| F7 | 106 | 6.6 % | 6.6 % | +0.0 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 3.3 %, with 3.3 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 12.0 % | 12.0 % | +0.0 | [+0.0, +0.0] |
| R2 | 612 | 0.5 % | 0.5 % | +0.0 | [+0.0, +0.0] |
| R3 | 589 | 5.3 % | 5.3 % | +0.0 | [+0.0, +0.0] |
| R4 | 495 | 1.6 % | 1.6 % | +0.0 | [+0.0, +0.0] |
| R5 | 549 | 5.5 % | 5.5 % | +0.0 | [+0.0, +0.0] |
| R6 | 28 | 3.6 % | 3.6 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 11.8 % | 11.8 % | +0.0 | [+0.0, +0.0] |
| threat faceswap | 272 | 78.3 % | 78.3 % | +0.0 | [+0.0, +0.0] |
| threat gan_face | 153 | 54.2 % | 54.2 % | +0.0 | [+0.0, +0.0] |
| threat personalised | 115 | 40.9 % | 40.9 % | +0.0 | [+0.0, +0.0] |
| threat t2i | 1603 | 3.9 % | 3.9 % | +0.0 | [+0.0, +0.0] |
| family faceswap-stills | 272 | 78.3 % | 78.3 % | +0.0 | [+0.0, +0.0] |
| family flux1 | 85 | 1.2 % | 1.2 % | +0.0 | [+0.0, +0.0] |
| family flux2 | 70 | 5.7 % | 5.7 % | +0.0 | [+0.0, +0.0] |
| family gan-faces | 153 | 54.2 % | 54.2 % | +0.0 | [+0.0, +0.0] |
| family gpt-4o-image-1 | 41 | 2.4 % | 2.4 % | +0.0 | [+0.0, +0.0] |
| family gpt-image-1.5-2 | 57 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| family hidream | 50 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| family hunyuanimage | 24 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 2.1 % | 2.1 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 53 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| family imagen4 | 106 | 7.5 % | 7.5 % | +0.0 | [+0.0, +0.0] |
| family legacy | 148 | 2.7 % | 2.7 % | +0.0 | [+0.0, +0.0] |
| family long-tail | 186 | 7.0 % | 7.0 % | +0.0 | [+0.0, +0.0] |
| family midjourney | 153 | 5.9 % | 5.9 % | +0.0 | [+0.0, +0.0] |
| family modern-t2i-faces | 115 | 7.0 % | 7.0 % | +0.0 | [+0.0, +0.0] |
| family nano-banana | 118 | 0.8 % | 0.8 % | +0.0 | [+0.0, +0.0] |
| family partial-edits | 161 | 11.8 % | 11.8 % | +0.0 | [+0.0, +0.0] |
| family personalised | 115 | 40.9 % | 40.9 % | +0.0 | [+0.0, +0.0] |
| family qwen-image | 106 | 0.9 % | 0.9 % | +0.0 | [+0.0, +0.0] |
| family recraft | 54 | 9.3 % | 9.3 % | +0.0 | [+0.0, +0.0] |
| family sd3 | 75 | 2.7 % | 2.7 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 5.7 % | 5.7 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 3.8 % | 3.8 % | +0.0 | [+0.0, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
