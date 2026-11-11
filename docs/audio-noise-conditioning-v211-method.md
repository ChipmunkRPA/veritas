# Mild noise conditioning for speech detection

The [v208 river result](audio-demand-transfer-v208-result.md) lost all 21
gain-matched clean AI directions at 0 dB even though all 24 synthetic
clips passed the speech gate. Most inconclusive results already occurred
before the original-side stability check. Compare one mild signal
transform that can change the classifier response. Prior local results
provide reasons to keep the transform small. [GTCRN noise reduction](audio-noise-view-v4-evaluation.md)
lost useful calls. The [band-limited comparison](audio-specialists-v15-result.md)
did not improve directional detection. [Source separation](audio-separation-v14-development-evaluation.md)
also lost useful results. A recent fixed-classifier study found that
better listening scores did not reliably predict better spoof detection.
[Speech enhancement study](https://arxiv.org/html/2603.14767v1)

Reuse the complete v208 public aggregate, private guarded journals and
prepared source commitments. Keep all 48 planned source slots in
gain-matched clean, washroom 0 dB, river 0 dB and river +5 dB.
Only the recorded speech gate's complete 2.5–4 second mono 16 kHz
windows are eligible for a transformed score. For each condition and
origin class, replay the first admitted original raw score to an
absolute tolerance of 0.000001. Keep the recorded serving outcomes and
their original stability checks.

Use a centered 512-sample periodic Hann analysis window and 128-sample
hop with reflected edges. Estimate each frequency bin's noise power as
the 20th percentile of its power across the clip. Multiply each complex
bin by the square root of the greater of 0.25 and one minus the noise
estimate divided by observed power plus 0.000000000001. This retains
phase and limits attenuation to about 6 dB in each bin. Use normalized
overlap-add and remove the padding to keep every original sample index.
Use one fixed transform. Mark constant, effectively silent, identical,
or out-of-range transformed inputs unavailable. No new pretrained weights
are required. The existing voice weights and calibration remain pinned.

Score each usable transformed view once and apply the unchanged bundle
only to compare initial directional decisions. The transformed score is
uncalibrated for this input distribution. Report baseline and transformed
ranking on identical assessable cases, exact score ties, response movement,
and initial direction transitions. Report the recorded serving direction
separately. In a post-hoc development simulation, allow a transformed
initial AI direction to replace only a recorded inconclusive result.
Preserve recorded AI, original, and unavailable results. Report the paired
new false AI calls, AI recoveries and all four directional counts.

The transformed waveform is a comparison input. Keep the original WAV,
gate result and sample interval authoritative for evidence. An estimated
noise floor computed over the clip does not identify a foreground speaker,
the source of a sound, or a manipulated sub-interval. The comparison reuses
43 prepared foregrounds, a single synthetic collection and two fixed
background recordings with class-specific offsets. Any promising result
needs a fresh, grouped, multiple-generator and multiple-background
confirmation with both classes exposed to identical noise excerpts and
separate calibration before a serving or confidence change.
[Media-transformation evaluation design](https://arxiv.org/html/2605.09568v1)

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_noise_conditioning_v211 preflight
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_noise_conditioning_v211 freeze
for batch in 0 1 2 3; do
  PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_noise_conditioning_v211 run "$batch"
done
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_noise_conditioning_v211 readout
```

Workers run offline under the unchanged v108 neural guard. The readout
recomputes the transform and validates its commitments and guarded
score journals. Source and transformed audio, masks and individual
scores stay private. The public readout contains counts and artifact
commitments.
