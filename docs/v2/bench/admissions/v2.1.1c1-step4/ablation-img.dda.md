# Ablation: img.dda

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.dfdc_b7,img.hf.ateeqq,img.hf.dima806 GATES='img.dfdc_b7=has_face;img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.dda ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dfdc_b7, img.hf.ateeqq, img.hf.dima806
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (MSCOCO train2017)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3
- Operating point (D-53): reals scoring exactly τ count as flagged with weight q (the rank rule's share); served rule (s ≥ τ) weighted FPR: without 5.0 % (q 0.099), with 5.0 % (q 0.407)

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +28.8 pts, CI [+24.3, +33.3] |
| A3b | pass | family faceswap-stills, family gan-faces, family hidream, family hunyuanimage, family ideogram, family imagen3, family imagen4, family legacy, family long-tail, family midjourney, family modern-t2i-faces, family partial-edits, family personalised, family recraft, family sd3, family seedream3, family seedream4, threat edit, threat faceswap, threat gan_face, threat personalised, threat t2i |
| A4 | pass | pooled reals: Δ FPR -0.0 pts, CI [-1.5, +1.2] (blocks with a lower bound above +1.0, D-53); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): R6 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 23.5 % | 87.6 % | +64.1 |
| F2 | 70 | 37.1 % | 45.7 % | +8.6 |
| F3 | 57 | 31.6 % | 29.8 % | -1.8 |
| F4 | 118 | 32.2 % | 57.6 % | +25.4 |
| F5 | 80 | 31.2 % | 71.2 % | +40.0 |
| F6 | 106 | 28.3 % | 44.3 % | +16.0 |
| F7 | 106 | 23.6 % | 72.6 % | +49.1 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 29.7 %, with 58.4 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 3.1 % | 3.8 % | +0.7 | [-3.6, +2.7] |
| R2 | 612 | 3.6 % | 5.1 % | +1.5 | [-3.4, +4.3] |
| R3 | 589 | 9.3 % | 6.5 % | -2.9 | [-5.2, -0.5] |
| R4 | 493 | 7.3 % | 5.9 % | -1.4 | [-3.9, +1.0] |
| R5 | 548 | 1.5 % | 3.0 % | +1.5 | [+0.0, +3.3] |
| R6 | 28 | 7.1 % | 17.9 % | +10.7 | [+0.0, +25.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 15.5 % | 67.1 % | +51.6 | [+42.2, +61.5] |
| threat faceswap | 272 | 13.6 % | 57.7 % | +44.1 | [+33.2, +55.5] |
| threat gan_face | 153 | 0.7 % | 39.9 % | +39.2 | [+31.4, +47.1] |
| threat personalised | 115 | 7.8 % | 96.5 % | +88.7 | [+82.6, +93.9] |
| threat t2i | 1600 | 33.4 % | 67.2 % | +33.8 | [+30.4, +37.2] |
| family faceswap-stills | 272 | 13.6 % | 57.7 % | +44.1 | [+33.2, +55.5] |
| family flux1 | 85 | 56.5 % | 67.1 % | +10.6 | [-2.4, +23.6] |
| family flux2 | 70 | 40.0 % | 42.9 % | +2.9 | [-8.5, +14.3] |
| family gan-faces | 153 | 0.7 % | 39.9 % | +39.2 | [+31.4, +47.1] |
| family gpt-4o-image-1 | 41 | 39.0 % | 41.5 % | +2.4 | [-17.1, +22.0] |
| family gpt-image-1.5-2 | 57 | 36.8 % | 19.3 % | -17.5 | [-35.1, +0.0] |
| family hidream | 50 | 56.0 % | 86.0 % | +30.0 | [+11.3, +49.0] |
| family hunyuanimage | 24 | 58.3 % | 100.0 % | +41.7 | [+20.8, +62.5] |
| family ideogram | 47 | 25.5 % | 78.7 % | +53.2 | [+36.2, +68.6] |
| family imagen3 | 53 | 20.8 % | 56.6 % | +35.8 | [+20.8, +50.9] |
| family imagen4 | 106 | 27.4 % | 64.2 % | +36.8 | [+26.2, +47.5] |
| family legacy | 145 | 42.1 % | 91.0 % | +49.0 | [+40.7, +57.6] |
| family long-tail | 186 | 37.1 % | 71.0 % | +33.9 | [+22.4, +44.8] |
| family midjourney | 153 | 29.4 % | 81.7 % | +52.3 | [+44.2, +60.8] |
| family modern-t2i-faces | 115 | 2.6 % | 67.8 % | +65.2 | [+55.9, +74.5] |
| family nano-banana | 118 | 39.0 % | 45.8 % | +6.8 | [-3.4, +16.8] |
| family partial-edits | 161 | 15.5 % | 67.1 % | +51.6 | [+42.2, +61.5] |
| family personalised | 115 | 7.8 % | 96.5 % | +88.7 | [+82.6, +93.9] |
| family qwen-image | 106 | 30.2 % | 37.7 % | +7.5 | [-1.8, +17.0] |
| family recraft | 54 | 29.6 % | 87.0 % | +57.4 | [+42.9, +71.2] |
| family sd3 | 75 | 24.0 % | 98.7 % | +74.7 | [+63.2, +84.8] |
| family seedream3 | 35 | 28.6 % | 80.0 % | +51.4 | [+35.1, +68.6] |
| family seedream4 | 80 | 35.0 % | 61.3 % | +26.3 | [+11.5, +39.8] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
