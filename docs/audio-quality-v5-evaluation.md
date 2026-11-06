# Original-waveform quality calibration: development result

Decision: **reject**. The existing detector and serving calibration remain
unchanged. No fresh confirmation audio was scored.

The [predeclared v5 experiment](audio-quality-v5-development.md) completed all
720 consumed-development inputs: 690 assessable, 30 retaining their original
unassessed status, and zero extraction failures. Sixteen focused tests passed,
including gain invariance, waveform preservation, predictor metadata exclusion,
monotonicity, comparator-objective parity, held-out label isolation and refusal
of connected identities crossing folds. Frozen manifest SHA-256:
`ff49bcf9a27205f407f6dcea324256b70a2f157e2c41601addadd5bf7bf5c242`.

The table compares the fairly refitted original-only and quality-conditioned
development mappings at unchanged research operating points. Each condition
contains 60 authentic and 60 generated inputs. Inconclusive includes unassessed
inputs. These are not the current guarded service decisions.

| Condition | Correct, original / candidate | False AI, original / candidate | False original, original / candidate | Inconclusive, original / candidate |
|---|---:|---:|---:|---:|
| Clean | 115 / 115 | 0 / 0 | 0 / 0 | 5 / 5 |
| Noise, 0 dB | 82 / 63 | 1 / 1 | 12 / 2 | 25 / 54 |
| Noise, 5 dB | 96 / 82 | 0 / 0 | 9 / 3 | 15 / 35 |
| Noise, 15 dB | 108 / 107 | 0 / 0 | 4 / 0 | 8 / 13 |
| Music, 5 dB | 99 / 101 | 2 / 2 | 2 / 2 | 17 / 15 |
| Music, 15 dB | 110 / 111 | 0 / 0 | 0 / 2 | 10 / 7 |
| All 720 inputs | 610 / 579 | 3 / 3 | 27 / 9 | 80 / 129 |

Common-assessed, source/class-weighted Brier improved from 0.057960 to 0.052092
and log loss from 0.201118 to 0.176139. The paired Brier difference was -0.005868,
with conditional grouped interval [-0.011549, -0.000098]. This interval conditions
on consumed sources, reused backgrounds and fixed fold fits, excluding refitting
uncertainty. There are 120 original files and 60 connected source groups; the
720 variants are not independent originals.

The candidate failed three substantive gates: music-5 Brier worsened; music-15
gained two false-original calls; and noise-0 lost 19 correct conclusions rather
than gaining five. Music-5 gained only two correct conclusions, below its fixed
five-call requirement. Better pooled probability scores therefore did not
establish a useful, comparable-error replacement.

On the joint VCTK/DFADD voice panel, Brier improved from 0.063876 to 0.059077,
but correct conclusions declined from 388 to 380. The quality-only nuisance
diagnostic had pooled AUC 0.467 and no directional calls. That diagnostic does
not establish absence of channel, source, prompt or pretraining confounding.
The complete generator, source and background partitions, fixed fold parameters,
both error directions and operational reference are retained in the
[machine-readable result](audio-quality-v5-evaluation.json).

These findings suggest the selected observable properties mostly change
uncertainty rather than recover enough discriminative information; they do not
prove that a different quality method could never help. Any further method
requires a new frozen design. Do not change this experiment's gates, claim
fresh validation, or consume unused confirmation data to rescue this result.
No outside reviewer has independently validated it.
