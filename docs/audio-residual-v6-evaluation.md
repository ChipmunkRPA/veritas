# Frozen-representation speech head: development result

Decision: **reject this replacement under its frozen advancement criteria**.
Preserve the running detector and its calibration. No fresh confirmation audio
was scored and the new decision layer is not used in customer confidence.

The [v6 protocol](audio-residual-v6-development.md) was frozen before extraction.
All 720 consumed inputs completed: 690 assessed and 30 retaining their original
unassessed status, with zero new failures. Every assessed input reproduced the
original predictor's raw response within 1e-7; captured vectors reconstructed
the existing head response within 1e-6. The eight frozen code/protocol/registry
hashes were verified after extraction. Manifest SHA-256:
`574a2512e44e7270a585577e7f9bb3e0a2a58e39fbc3dd4e95b1bca26a7cd895`.

## Complete outcomes

These are held-out connected-fold development mappings at fixed research
operating points. The original-only comparator is fairly refitted; it is not
the currently deployed guarded calibration. Each condition has 60 genuine and
60 generated inputs. Inconclusive includes unassessed inputs.

| Condition | Correct, original / residual head | False AI, original / residual head | False original, original / residual head | Inconclusive, original / residual head |
|---|---:|---:|---:|---:|
| Clean | 115 / 115 | 0 / 0 | 0 / 0 | 5 / 5 |
| Noise, 0 dB | 82 / 76 | 1 / 2 | 12 / 2 | 25 / 40 |
| Noise, 5 dB | 96 / 94 | 0 / 0 | 9 / 1 | 15 / 25 |
| Noise, 15 dB | 108 / 106 | 0 / 0 | 4 / 0 | 8 / 14 |
| Music, 5 dB | 99 / 100 | 2 / 2 | 2 / 2 | 17 / 16 |
| Music, 15 dB | 110 / 110 | 0 / 0 | 0 / 0 | 10 / 10 |
| All 720 inputs | 610 / 601 | 3 / 4 | 27 / 5 | 80 / 110 |

Source/class-weighted Brier on common assessed inputs improved from 0.057960 to
0.038711 and log loss from 0.201118 to 0.137161. The paired Brier difference was
-0.019249, with conditional grouped interval [-0.028627, -0.009396]. Every
condition improved Brier. The shared-voice panel also improved Brier, from
0.063876 to 0.049515. Probability scoring improved more than in the v4/v5
experiments, suggesting information remains upstream of the original single
response. This is a development inference, not a mechanism or generalization
claim.

However, the original fixed usefulness gate required at least five additional
correct conclusions in both noise-0 and music-5. The residual head lost six in
noise-0 and gained only one in music-5. All other fixed gates passed. The
stronger proper scores and fewer false-original calls do not erase this failed
criterion, nor establish success on unknown generators. The method is not
advanced to fresh confirmation by changing the rule after observing outcomes.

The unchanged guarded service reference on these consumed inputs made 500
correct conclusions, one false AI call, no false-original calls and 219
inconclusive results. This is separately reported in the
[complete machine-readable record](audio-residual-v6-evaluation.json), alongside
generator, source and background partitions. Do not attribute the refitted
comparator's 27 false-original calls to the running service.

## Reliability and next decision

PCA, centering, scaling and fitting used training folds only; neither held-out
representations nor labels entered their own fit. Tests explicitly altered
held-out vectors/labels and checked unchanged corresponding fitted parameters.
The backbone, layer mixture and input preparation remained frozen; only a small
new decision layer was learned. Current customer reports and public names did
not change.

There are 120 original files and 60 connected groups, not 720 independent
observations. Intervals condition on the reused background pool and fixed fold
fits, excluding fit uncertainty. Source/channel confounding, undisclosed exact
DFADD prompts and unknown pretrained overlap remain. No outside independent
reviewer has evaluated these outcomes.

Further audio work needs separately specified fitting/calibration/confirmation
boundaries and an uncertainty policy for a new decision layer, evaluated against
a matched guarded baseline. It must not inherit the old mapping or quietly relax
this failed experiment's gate. Broader background-speech and new-generator
confirmation remains outstanding.
