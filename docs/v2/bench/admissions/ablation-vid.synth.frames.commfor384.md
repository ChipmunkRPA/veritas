# Ablation: vid.synth.frames.commfor384

- Command: `make calibrate MODALITY=video SEED=7 BENCH=vb-1.0-core DETECTORS=vid.frames.univfd,vid.frames.hf.ateeqq,vid.frames.hf.dima806,vid.frames.dfdc_b7,vid.codec.ffprobe,vid.lipsync.landmark_xcorr,aud.gustking_xlsr,vid.synth.frames.commfor384 ABLATE=vid.synth.frames.commfor384 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `d0224343a5c0ad38605ade2b5225deba3cb607ace950e2ae3a127ba9dab26d25`
- Baseline detectors: aud.gustking_xlsr, vid.codec.ffprobe, vid.frames.dfdc_b7, vid.frames.hf.ateeqq, vid.frames.hf.dima806, vid.frames.univfd, vid.lipsync.landmark_xcorr
- Items (fit ∪ cal): 80 (37 real, 43 fake); contaminated dropped 0 (OwensLab/CommunityForensics-Eval)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro +0.0 pts, CI [+0.0, +0.0] |
| A3b | pass | family stable-video, family videopoet, threat t2v |
| A4 | fail | failing strata: V-general |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| V-Kling | 3 | 0.0 % | 0.0 % | +0.0 |
| V-Seedance | 3 | 0.0 % | 0.0 % | +0.0 |
| V-Sora | 3 | 0.0 % | 0.0 % | +0.0 |
| V-Veo | 2 | 0.0 % | 0.0 % | +0.0 |
| V-WanHunyuan | 2 | 0.0 % | 0.0 % | +0.0 |

Macro over V-Kling, V-Seedance, V-Sora, V-Veo, V-WanHunyuan: without 0.0 %, with 0.0 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| V-general | 30 | 3.3 % | 3.3 % | +0.0 | [-10.0, +10.0] |
| V-talking-head | 7 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat t2v | 41 | 7.3 % | 41.5 % | +34.1 | [+18.4, +51.3] |
| threat talking_head | 2 | 0.0 % | 50.0 % | +50.0 | [+0.0, +100.0] |
| family ai-talking-humans | 3 | 0.0 % | 66.7 % | +66.7 | [+0.0, +100.0] |
| family animatediff | 3 | 0.0 % | 66.7 % | +66.7 | [+0.0, +100.0] |
| family cogvideox | 3 | 33.3 % | 0.0 % | -33.3 | [-100.0, +0.0] |
| family kling | 3 | 0.0 % | 33.3 % | +33.3 | [+0.0, +100.0] |
| family luma-ray | 3 | 33.3 % | 0.0 % | -33.3 | [-100.0, +0.0] |
| family marey | 3 | 0.0 % | 33.3 % | +33.3 | [+0.0, +100.0] |
| family mochi | 3 | 0.0 % | 33.3 % | +33.3 | [+0.0, +100.0] |
| family pika | 3 | 33.3 % | 100.0 % | +66.7 | [+0.0, +100.0] |
| family runway | 3 | 0.0 % | 66.7 % | +66.7 | [+0.0, +100.0] |
| family seedance | 3 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| family sora | 3 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| family stable-video | 2 | 0.0 % | 100.0 % | +100.0 | [+100.0, +100.0] |
| family talking-head-animation | 2 | 0.0 % | 50.0 % | +50.0 | [+0.0, +100.0] |
| family veo | 2 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| family videopoet | 2 | 0.0 % | 100.0 % | +100.0 | [+100.0, +100.0] |
| family wan-hunyuan | 2 | 0.0 % | 50.0 % | +50.0 | [+0.0, +100.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
