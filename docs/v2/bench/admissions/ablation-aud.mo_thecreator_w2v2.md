# Ablation: aud.mo_thecreator_w2v2

- Command: `make calibrate MODALITY=audio SEED=7 BENCH=vb-1.0-core DETECTORS=aud.gustking_xlsr,aud.mo_thecreator_w2v2 ABLATE=aud.mo_thecreator_w2v2 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `d0224343a5c0ad38605ade2b5225deba3cb607ace950e2ae3a127ba9dab26d25`
- Baseline detectors: aud.gustking_xlsr
- Items (fit ∪ cal): 149 (75 real, 74 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | not established | Δ LGO macro – pts, CI – |
| A3b | not established | no family or threat gains ≥ 10 pts |
| A4 | fail | failing strata: A-bonafide |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|

Macro over –: without –, with –. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| A-bonafide | 75 | 4.0 % | 4.0 % | +0.0 | [-4.1, +4.2] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat tts | 74 | 21.6 % | 13.5 % | -8.1 | [-16.7, +0.0] |
| family asvspoof2021-df | 19 | 26.3 % | 15.8 % | -10.5 | [-31.6, +10.5] |
| family asvspoof5 | 19 | 52.6 % | 31.6 % | -21.1 | [-47.4, +0.0] |
| family in-the-wild-audio | 18 | 5.6 % | 5.6 % | +0.0 | [+0.0, +0.0] |
| family mlaad | 18 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
