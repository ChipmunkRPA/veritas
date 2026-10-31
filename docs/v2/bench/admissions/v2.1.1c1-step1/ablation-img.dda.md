# Ablation: img.dda

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=face.gend_clip_l14,img.commfor384,img.dda,img.dfdc_b7,img.hf.ateeqq,img.hf.dima806 GATES='face.gend_clip_l14=has_face;img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.dda ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: face.gend_clip_l14, img.commfor384, img.dfdc_b7, img.hf.ateeqq, img.hf.dima806
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (MSCOCO train2017)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3
- Operating point (D-53): reals scoring exactly τ count as flagged with weight q (the rank rule's share); served rule (s ≥ τ) weighted FPR: without 5.0 % (q 0.331), with 5.0 % (q 0.413)

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +14.5 pts, CI [+11.7, +17.6] |
| A3b | pass | family flux1, family flux2, family gpt-4o-image-1, family hidream, family ideogram, family imagen3, family imagen4, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family qwen-image, family recraft, family sd3, family seedream3, family seedream4, threat t2i |
| A4 | pass | pooled reals: Δ FPR +0.1 pts, CI [-0.6, +0.7] (blocks with a lower bound above +1.0, D-53); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R5; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 68.0 % | 81.0 % | +13.1 |
| F2 | 70 | 25.7 % | 38.6 % | +12.9 |
| F3 | 57 | 10.5 % | 14.0 % | +3.5 |
| F4 | 118 | 27.1 % | 40.7 % | +13.6 |
| F5 | 80 | 63.7 % | 76.2 % | +12.5 |
| F6 | 106 | 26.4 % | 40.6 % | +14.2 |
| F7 | 106 | 31.1 % | 63.2 % | +32.1 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 36.1 %, with 50.6 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 8.1 % | 7.3 % | -0.8 | [-2.9, +0.7] |
| R2 | 612 | 3.3 % | 3.4 % | +0.2 | [-1.8, +1.5] |
| R3 | 589 | 4.6 % | 4.4 % | -0.2 | [-1.5, +1.2] |
| R4 | 493 | 6.1 % | 4.1 % | -2.0 | [-3.7, -0.4] |
| R5 | 548 | 2.4 % | 5.2 % | +2.8 | [+1.4, +4.4] |
| R6 | 28 | 10.7 % | 10.7 % | +0.0 | [-15.4, +14.3] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 57.1 % | 65.2 % | +8.1 | [+3.7, +13.0] |
| threat faceswap | 272 | 77.9 % | 79.8 % | +1.8 | [+0.0, +4.3] |
| threat gan_face | 153 | 55.6 % | 59.5 % | +3.9 | [+0.0, +7.9] |
| threat personalised | 115 | 90.4 % | 93.9 % | +3.5 | [+0.9, +7.0] |
| threat t2i | 1600 | 46.9 % | 65.6 % | +18.6 | [+16.5, +20.7] |
| family faceswap-stills | 272 | 77.9 % | 79.8 % | +1.8 | [+0.0, +4.3] |
| family flux1 | 85 | 45.9 % | 64.7 % | +18.8 | [+8.4, +28.1] |
| family flux2 | 70 | 24.3 % | 37.1 % | +12.9 | [+5.6, +21.1] |
| family gan-faces | 153 | 55.6 % | 59.5 % | +3.9 | [+0.0, +7.9] |
| family gpt-4o-image-1 | 41 | 0.0 % | 17.1 % | +17.1 | [+7.3, +29.3] |
| family gpt-image-1.5-2 | 57 | 8.8 % | 12.3 % | +3.5 | [-3.5, +12.3] |
| family hidream | 50 | 70.0 % | 86.0 % | +16.0 | [+6.1, +26.5] |
| family hunyuanimage | 24 | 95.8 % | 100.0 % | +4.2 | [+0.0, +12.5] |
| family ideogram | 47 | 48.9 % | 72.3 % | +23.4 | [+12.0, +36.4] |
| family imagen3 | 53 | 26.4 % | 50.9 % | +24.5 | [+11.3, +37.7] |
| family imagen4 | 106 | 25.5 % | 61.3 % | +35.8 | [+26.4, +45.7] |
| family legacy | 145 | 93.1 % | 94.5 % | +1.4 | [-1.3, +4.5] |
| family long-tail | 186 | 50.5 % | 70.4 % | +19.9 | [+13.2, +26.8] |
| family midjourney | 153 | 64.7 % | 81.7 % | +17.0 | [+11.3, +22.7] |
| family modern-t2i-faces | 115 | 36.5 % | 64.3 % | +27.8 | [+20.5, +35.0] |
| family nano-banana | 118 | 25.4 % | 40.7 % | +15.3 | [+8.5, +22.9] |
| family partial-edits | 161 | 57.1 % | 65.2 % | +8.1 | [+3.7, +13.0] |
| family personalised | 115 | 90.4 % | 93.9 % | +3.5 | [+0.9, +7.0] |
| family qwen-image | 106 | 24.5 % | 37.7 % | +13.2 | [+5.9, +20.8] |
| family recraft | 54 | 48.1 % | 81.5 % | +33.3 | [+20.4, +48.1] |
| family sd3 | 75 | 65.3 % | 98.7 % | +33.3 | [+22.7, +44.0] |
| family seedream3 | 35 | 48.6 % | 82.9 % | +34.3 | [+17.6, +50.0] |
| family seedream4 | 80 | 62.5 % | 73.8 % | +11.3 | [+2.5, +20.7] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
