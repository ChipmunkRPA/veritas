# Ablation: vid.synth.frames.commfor384

- Command: `make calibrate MODALITY=video SEED=7 BENCH=vb-1.1 DETECTORS=vid.synth.frames.commfor384,vid.synth.frames.dda,vid.frames.dfdc_b7,vid.codec.ffprobe,aud.gustking_xlsr,vid.lipsync.landmark_xcorr CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=vid.synth.frames.commfor384 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: aud.gustking_xlsr, vid.codec.ffprobe, vid.frames.dfdc_b7, vid.lipsync.landmark_xcorr, vid.synth.frames.dda
- Items (fit ∪ cal): 353 (188 real, 165 fake); contaminated dropped 0 (OwensLab/CommunityForensics-Eval)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro +4.0 pts, CI [-8.0, +16.8] |
| A3b | pass | family ai-talking-humans, family kling, family seedance, family talking-head-animation, threat t2v, threat talking_head |
| A4 | fail | pooled reals: Δ FPR +0.0 pts, CI [-3.7, +3.7] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: V-general, V-talking-head; point above +2.0 without established harm: none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| V-Kling | 8 | 37.5 % | 62.5 % | +25.0 |
| V-Seedance | 11 | 27.3 % | 36.4 % | +9.1 |
| V-Sora | 12 | 33.3 % | 41.7 % | +8.3 |
| V-Veo | 9 | 88.9 % | 66.7 % | -22.2 |
| V-WanHunyuan | 10 | 100.0 % | 100.0 % | +0.0 |

Macro over V-Kling, V-Seedance, V-Sora, V-Veo, V-WanHunyuan: without 57.4 %, with 61.4 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| V-general | 182 | 4.4 % | 4.9 % | +0.5 | [-2.7, +4.4] |
| V-talking-head | 6 | 16.7 % | 0.0 % | -16.7 | [-37.5, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat t2v | 157 | 73.9 % | 88.5 % | +14.6 | [+8.8, +21.1] |
| threat talking_head | 8 | 25.0 % | 75.0 % | +50.0 | [+12.5, +87.5] |
| family ai-talking-humans | 12 | 58.3 % | 91.7 % | +33.3 | [+8.3, +58.3] |
| family animatediff | 10 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family cogvideox | 11 | 90.9 % | 100.0 % | +9.1 | [+0.0, +27.3] |
| family kling | 8 | 25.0 % | 87.5 % | +62.5 | [+25.0, +100.0] |
| family luma-ray | 12 | 83.3 % | 91.7 % | +8.3 | [-18.2, +40.0] |
| family marey | 11 | 72.7 % | 100.0 % | +27.3 | [+0.0, +54.5] |
| family mochi | 10 | 90.0 % | 100.0 % | +10.0 | [+0.0, +30.0] |
| family pika | 11 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family runway | 10 | 90.0 % | 100.0 % | +10.0 | [+0.0, +30.0] |
| family seedance | 11 | 0.0 % | 45.5 % | +45.5 | [+18.2, +72.7] |
| family sora | 12 | 41.7 % | 50.0 % | +8.3 | [+0.0, +25.0] |
| family stable-video | 11 | 90.9 % | 100.0 % | +9.1 | [+0.0, +27.3] |
| family talking-head-animation | 8 | 25.0 % | 75.0 % | +50.0 | [+12.5, +87.5] |
| family veo | 9 | 77.8 % | 66.7 % | -11.1 | [-44.4, +22.2] |
| family videopoet | 9 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family wan-hunyuan | 10 | 90.0 % | 100.0 % | +10.0 | [+0.0, +30.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
