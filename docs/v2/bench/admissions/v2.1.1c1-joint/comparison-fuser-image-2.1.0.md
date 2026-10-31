# Comparison: fuser-image-2.1.0

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.dfdc_b7,img.hf.ateeqq,img.hf.dima806 GATES='img.dfdc_b7=has_face;img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet REFERENCE=img.commfor384,img.dda,img.dfdc_b7 REFERENCE_GATES='img.dfdc_b7=has_face' REFERENCE_LABEL=fuser-image-2.1.0 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- With: img.dda, img.dfdc_b7 (has_face), img.hf.ateeqq (!has_face), img.hf.dima806
- Without: img.commfor384, img.dda, img.dfdc_b7 (has_face)
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (OwensLab/CommunityForensics-Eval)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3
- Operating point (D-53): reals scoring exactly τ count as flagged with weight q (the rank rule's share); served rule (s ≥ τ) weighted FPR: without 5.3 % (q 0.470), with 5.0 % (q 0.407)

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +14.3 pts, CI [+10.7, +17.8] |
| A3b | pass | family flux2, family gpt-4o-image-1, family imagen3, family imagen4, family modern-t2i-faces, family nano-banana |
| A4 | pass | pooled reals: Δ FPR +0.1 pts, CI [-0.9, +1.1] (blocks with a lower bound above +1.0, D-53); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R1, R3; above +2.0 on fewer than 150 reals (no term): R6 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 81.0 % | 87.6 % | +6.5 |
| F2 | 70 | 25.7 % | 45.7 % | +20.0 |
| F3 | 57 | 14.0 % | 29.8 % | +15.8 |
| F4 | 118 | 33.1 % | 57.6 % | +24.6 |
| F5 | 80 | 75.0 % | 71.2 % | -3.7 |
| F6 | 106 | 29.2 % | 44.3 % | +15.1 |
| F7 | 106 | 50.9 % | 72.6 % | +21.7 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 44.1 %, with 58.4 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 1.4 % | 3.8 % | +2.4 | [-0.6, +3.9] |
| R2 | 612 | 4.7 % | 5.1 % | +0.3 | [-1.1, +1.6] |
| R3 | 589 | 3.6 % | 6.5 % | +2.9 | [+1.0, +4.9] |
| R4 | 493 | 13.6 % | 5.9 % | -7.7 | [-10.8, -4.9] |
| R5 | 548 | 1.4 % | 3.0 % | +1.6 | [+0.3, +3.1] |
| R6 | 28 | 3.6 % | 17.9 % | +14.3 | [+3.3, +29.2] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 63.4 % | 67.1 % | +3.7 | [-0.6, +8.7] |
| threat faceswap | 272 | 62.9 % | 57.7 % | -5.1 | [-10.2, -1.1] |
| threat gan_face | 153 | 53.6 % | 39.9 % | -13.7 | [-23.5, -3.9] |
| threat personalised | 115 | 96.5 % | 96.5 % | +0.0 | [-3.5, +3.5] |
| threat t2i | 1600 | 61.0 % | 67.2 % | +6.3 | [+4.4, +8.1] |
| family faceswap-stills | 272 | 62.9 % | 57.7 % | -5.1 | [-10.2, -1.1] |
| family flux1 | 85 | 61.2 % | 67.1 % | +5.9 | [+0.0, +13.3] |
| family flux2 | 70 | 27.1 % | 42.9 % | +15.7 | [+5.9, +26.0] |
| family gan-faces | 153 | 53.6 % | 39.9 % | -13.7 | [-23.5, -3.9] |
| family gpt-4o-image-1 | 41 | 7.3 % | 41.5 % | +34.1 | [+19.5, +48.8] |
| family gpt-image-1.5-2 | 57 | 14.0 % | 19.3 % | +5.3 | [-1.8, +12.3] |
| family hidream | 50 | 80.0 % | 86.0 % | +6.0 | [-4.2, +16.3] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 74.5 % | 78.7 % | +4.3 | [-6.1, +15.2] |
| family imagen3 | 53 | 41.5 % | 56.6 % | +15.1 | [+1.9, +28.3] |
| family imagen4 | 106 | 50.9 % | 64.2 % | +13.2 | [+4.7, +21.5] |
| family legacy | 145 | 95.1 % | 91.0 % | -4.1 | [-8.7, +0.5] |
| family long-tail | 186 | 63.4 % | 71.0 % | +7.5 | [+2.8, +12.6] |
| family midjourney | 153 | 81.7 % | 81.7 % | +0.0 | [-4.6, +4.7] |
| family modern-t2i-faces | 115 | 54.8 % | 67.8 % | +13.0 | [+8.3, +18.5] |
| family nano-banana | 118 | 32.2 % | 45.8 % | +13.6 | [+5.1, +21.8] |
| family partial-edits | 161 | 63.4 % | 67.1 % | +3.7 | [-0.6, +8.7] |
| family personalised | 115 | 96.5 % | 96.5 % | +0.0 | [-3.5, +3.5] |
| family qwen-image | 106 | 29.2 % | 37.7 % | +8.5 | [+1.0, +16.0] |
| family recraft | 54 | 87.0 % | 87.0 % | +0.0 | [-10.7, +9.6] |
| family sd3 | 75 | 97.3 % | 98.7 % | +1.3 | [+0.0, +4.1] |
| family seedream3 | 35 | 74.3 % | 80.0 % | +5.7 | [-8.6, +20.0] |
| family seedream4 | 80 | 75.0 % | 61.3 % | -13.7 | [-23.1, -6.1] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
