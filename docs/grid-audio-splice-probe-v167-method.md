# Ordinary speech-copy boundary probe

The full GRID serving pilot produced false AI tendencies on 12 of 16 genuine
soundtracks with an ordinary local 320 ms copy. None of the 16 untouched
originals leaned AI. This probe tests how the same voice check responds to
smoothing the copy boundaries and how a symmetric stability check changes its
decisions. It uses the 16 already consumed GRID pilot recordings, across eight
connected source components. No reserve source is used.
[Serving result](grid-serving-completion-result.md)

The original and hard-copy arms must match the exact historical audio. Two
additional arms apply 20 ms and 50 ms raised-cosine fades inside the same
output interval, mixing the original and the same copied samples at both
ends. The middle of the copy is unchanged by the fade. Round once to the
signed-16 lattice. Length, sample origin and outside samples stay fixed.
Verify that the sample jump at each immediate boundary matches the untouched
original after fading. These comparisons can reveal sensitivity to the joint
and its nearby context. Smoothing also changes the local waveform.
[Matched editing interventions](https://arxiv.org/html/2610.04479v1#S3.SS2) · [Splicing-artifact analysis](https://www.isca-archive.org/asvspoof_2024/negroni24_asvspoof.pdf)

Keep the production speech gate, full-profile window selection, raw mean and
existing local calibration. Bind all historical and new PCM before inference.
The first two arms must reproduce each stored availability, exact selected
sample window, voice direction and raw response within 0.000001. Both views
use the same source windows. The second view uses the existing whole-clip
100–3,800 Hz bandpass sensitivity transform. Reapply the current stability
limit symmetrically as a separate diagnostic. An AI tendency whose filtered
response is unavailable or changes the rounded raw log odds by more than
`log(2)` becomes inconclusive. Existing original-side rules still apply.
Neither the band-view raw score nor the difference supplies a second vote.

The separate PCM cue searches for exact nonoverlapping repeated ranges at
variable lags and records only the observed sample intervals. The 320 ms
copy overwrites part of its own donor. Only the surviving donor and the
matching portion of the output are directly comparable. A fade can shorten
this exact witness. The cue carries no speech-origin vote or inference
about the full copied interval. Missing, quiet, or highly repetitive input
can leave the measurement unavailable or ambiguous.
[Audio copy-move research](https://arxiv.org/abs/2302.07584)

Report planned, unassessed, inconclusive and directional outcomes for each
arm, plus paired raw changes against the hard copy. All inputs contain only
genuine audio. The study can measure these false AI tendencies. Missed AI
speech and performance on independent recordings require a separate study
with genuine and AI controls. No parameters or customer conclusions change.
The published result contains aggregates and whole-artifact commitments.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.grid_audio_splice_probe_v167 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.grid_audio_splice_probe_v167 run
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.grid_audio_splice_probe_v167 readout
```

The runner verifies the existing commercially reviewed voice and
speech-activity assets, uses CPU float32 and the prior strict mapped-weight
load, and records actual root voice forwards. The owned resource guard keeps
all attempted logs and profiles.
