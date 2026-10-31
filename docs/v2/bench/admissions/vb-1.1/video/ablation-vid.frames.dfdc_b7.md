# Ablation: vid.frames.dfdc_b7

- Command: `make calibrate MODALITY=video SEED=7 BENCH=vb-1.1 DETECTORS=vid.synth.frames.commfor384,vid.synth.frames.dda,vid.frames.dfdc_b7,vid.codec.ffprobe,aud.gustking_xlsr,vid.lipsync.landmark_xcorr CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=vid.frames.dfdc_b7 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: aud.gustking_xlsr, vid.codec.ffprobe, vid.lipsync.landmark_xcorr, vid.synth.frames.commfor384, vid.synth.frames.dda
- Items (fit ∪ cal): 353 (188 real, 165 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -2.5 pts, CI [-8.0, +0.0] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | fail | pooled reals: Δ FPR +0.0 pts, CI [-1.6, +1.6] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: V-general, V-talking-head; point above +2.0 without established harm: none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| V-Kling | 8 | 75.0 % | 62.5 % | -12.5 |
| V-Seedance | 11 | 36.4 % | 36.4 % | +0.0 |
| V-Sora | 12 | 41.7 % | 41.7 % | +0.0 |
| V-Veo | 9 | 66.7 % | 66.7 % | +0.0 |
| V-WanHunyuan | 10 | 100.0 % | 100.0 % | +0.0 |

Macro over V-Kling, V-Seedance, V-Sora, V-Veo, V-WanHunyuan: without 63.9 %, with 61.4 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| V-general | 182 | 4.9 % | 4.9 % | +0.0 | [-1.6, +1.6] |
| V-talking-head | 6 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat t2v | 157 | 87.9 % | 88.5 % | +0.6 | [+0.0, +1.9] |
| threat talking_head | 8 | 75.0 % | 75.0 % | +0.0 | [+0.0, +0.0] |
| family ai-talking-humans | 12 | 91.7 % | 91.7 % | +0.0 | [+0.0, +0.0] |
| family animatediff | 10 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family cogvideox | 11 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family kling | 8 | 87.5 % | 87.5 % | +0.0 | [+0.0, +0.0] |
| family luma-ray | 12 | 83.3 % | 91.7 % | +8.3 | [+0.0, +23.1] |
| family marey | 11 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family mochi | 10 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family pika | 11 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family runway | 10 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family seedance | 11 | 45.5 % | 45.5 % | +0.0 | [+0.0, +0.0] |
| family sora | 12 | 50.0 % | 50.0 % | +0.0 | [+0.0, +0.0] |
| family stable-video | 11 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family talking-head-animation | 8 | 75.0 % | 75.0 % | +0.0 | [+0.0, +0.0] |
| family veo | 9 | 66.7 % | 66.7 % | +0.0 | [+0.0, +0.0] |
| family videopoet | 9 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family wan-hunyuan | 10 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
