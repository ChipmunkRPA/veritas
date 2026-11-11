# Mild spectral conditioning failed to recover AI speech flags

All four local batches completed under the frozen resource guard. The
workers scored 169 transformed inputs and replayed eight untransformed
scores. All eight matched the earlier scores within the fixed tolerance.
The test kept the original speech-gate windows and source timestamps.
[Method](audio-noise-conditioning-v211-method.md) · [Verified counts](evaluation-panels/audio-noise-conditioning-complete-v211.json)

| Condition | Recorded AI flags out of 24 synthetic clips | AI flags after conditioning | Raw-score ranking AUC, recorded → conditioned |
| --- | ---: | ---: | ---: |
| Matched clean | 21 | 21 | 0.998 → 0.991 |
| Washroom, 0 dB | 15 | 14 | 1.000 → 0.995 |
| River, 0 dB | 0 | 0 | 0.838 → 0.764 |
| River, +5 dB | 4 | 3 | 0.940 → 0.898 |

Conditioned directions are initial diagnostic decisions under the original
mapping. In the separate rescue simulation, a conditioned AI direction
could replace an inconclusive result. The rule recovered zero AI flags
and added zero genuine false alarms. At river 0 dB, the conditioned view
initially put five synthetic clips on the original side. The recorded
route's final direction leaned original on two after its stability check.

Drop this transform from the replacement candidates. The comparison
reuses the previously assessed Turkish foregrounds and two recordings.
The fresh comparison uses shared noise excerpts across classes and
includes a second generator. The river crossover first measures the
effect of sharing the early and late excerpts on the original voice route.
