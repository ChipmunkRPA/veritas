# Keeping a common spectrogram scale across a noisy speech clip

A short speech edit can look different depending on how much competing speech
is in each analysis window. This comparison tests whether sharing the
normalization scale across the submitted clip helps locate the edit. It
retains the original mixture, speech-activity decisions and classifier.
[Segment-aware localization](https://arxiv.org/abs/2601.21925) evaluates
the interior of a manipulated interval separately from its edges. This
implementation uses that distinction and the project's existing frame
readout. It uses no assets from the paper.

## Fixed comparison

- Score the first and last four seconds of each six-second 16 kHz mono
  input. Keep the local mean and variance, share the variance only, or
  share both moments. The shared reference takes the first three seconds
  from the first view and the final three seconds from the second, so the
  two-second overlap never counts twice. Each part contributes variance
  around its own mean, which removes the simple between-part level offset
  from the variance calculation.
- Use the same one-second candidate windows, original half-window speech
  gate and affine frame head in all three comparisons. Use the median
  response inside each candidate and the lower median when both contexts
  cover it. Verify the local normalization against the existing frame
  calculation. The model can still use up to four seconds of context,
  including speech outside the indicated range.
- Calculate a separate raw point for each route using only the
  44-recording stock-voice development bank. Each point is the larger of
  the highest new room-only response and the second-highest new ordinary
  group response. Apply the comparison strictly and retain the existing
  short-review flags in the supplement arm. A useful location covers at
  least half the true edit with temporal intersection over union of at
  least 0.30. Report flags outside that overlap separately.

The three comparison banks had prior project exposure. The randomized
stock-voice bank includes two checkpoints, the three-generator bank
adds Chatterbox, ElevenLabs v4 and Kokoro, and the SINE bank measures
shorter publisher-labeled splices and same-word resynthesis with recorded
competing speech. These are multiple generator and speaker groups.
They do not establish real-world accuracy or performance on previously
unseen source families. Each record pins the submitted PCM and earlier
decision. The experiment uses the existing locally installed assets
under the publisher grants recorded in the project audit.
[Randomized placement and source terms](audio-random-clock-stock-2026-11-13.md)
· [Three-generator source panel](audio-remote-neighbor-replication-2026-11-12.md)
· [SINE and AMI source terms](audio-sine-resynthesis-transfer-2026-11-14.md)
· [Installed asset terms](commercial-model-audit.md)

## Results

### Noise with stock voices and different generators

Each cell lists useful locations, misplaced flags, and ordinary-file flags,
in that order. The current review and the variants flagged no isolated
background in either panel.

| Route | Randomized stock voices (16 edits, 24 ordinary) | Three-generator bank (12 edits, 36 ordinary) |
| --- | ---: | ---: |
| Current short review | 1 / 1 / 3 | 0 / 2 / 5 |
| Dense windows with local normalization | 4 / 1 / 6 | 2 / 4 / 11 |
| Shared scale alone | 5 / 1 / 5 | 2 / 4 / 11 |
| Shared center and scale alone | 5 / 2 / 8 | 2 / 5 / 17 |
| Current review plus shared scale | 5 / 1 / 6 | 2 / 5 / 14 |

The shared scale added one useful location over the dense local comparator
in randomized stock voices. It marked **4.500–5.500 s** for a stock-voice
replacement at **4.980–5.940 s**, alongside an already located edit from
the other generator in the same host. Although flags fell from six
ordinary files to five between the dense local and shared-scale arms,
the five spanned five source groups instead of four. In the three-generator
bank, shared scale and dense local found the same Chatterbox and Kokoro
locations and flagged 11 ordinary files each. Supplementing the current
review with shared scale added three ordinary groups in the stock bank and
five in the three-generator bank.

A retrospective sweep shows how that tradeoff changes with the number of
ordinary groups flagged. With the same two-group count as the current
review, shared scale found one of 16 stock edits and one of 12 edits in
the three-generator bank. At three groups it found four and two,
respectively, with three stock and nine three-generator ordinary file
flags. These points use held control scores and are descriptive.

### Shorter edits and unchanged words

The third comparison includes 12 short splices, 24 untouched clips, 12
deliveries with the same words resynthesized, and eight isolated rooms.
The splice durations span 0.28–0.77 seconds. An existing two-second
window can meet the 0.30 overlap requirement only on edits at least
0.60 seconds long. The one-second candidate windows improve that geometric
limit independently of normalization.

| Route | Useful splice locations | Misplaced splice flags | Untouched clips flagged | Same words resynthesized and flagged |
| --- | ---: | ---: | ---: | ---: |
| Current short review | 0/12 | 1/12 | 0/24 | 1/12 |
| Dense windows with local normalization | 1/12 | 1/12 | 4/24 | 4/12 |
| Shared scale alone | 1/12 | 1/12 | 5/24 | 4/12 |
| Shared center and scale alone | 1/12 | 1/12 | 5/24 | 5/12 |
| Current review plus shared scale | 0/12 | 2/12 | 5/24 | 4/12 |

Every dense route found the same **1.510–1.850 s** splice and used
**1.500–2.500 s** as its displayed range. The current review flagged
**1.000–3.000 s** in that same delivery. Preserving the existing range
prevented the supplement from making the timing narrower. All isolated
rooms remained unflagged. The existing review left four rooms unassessed
and the dense routes left three. Every speech delivery was assessed.

Across the three banks, standalone shared scale located 8 of 40 edits
and flagged 21 of 84 ordinary files. The current review located 1 of 40
and flagged 8 of 84. Denser windows with local normalization accounted
for seven of the eight locations and flagged 21 of 84 ordinary files.
The ordinary flags are matched deliveries across repeated source groups.
The new method remains experimental. The next comparison to settle is
shared scale versus dense local at a prospectively fixed false-alarm
point on new speakers and distinct generator families. A separate
timing test can determine whether replacing an already flagged broad
interval with a corroborated narrow interval improves the evidence.

## Replay

The [score ledger](evaluation-panels/audio-shared-mel-context-2026-11-14.json)
contains timing, numeric responses, and SHA-256 commitments. It contains
no audio or checkpoints. The review recomputes the separate operating
points and all panel counts with standard-library Python.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.audio_shared_mel_context review \
  docs/evaluation-panels/audio-shared-mel-context-2026-11-14.json
```

Native replay uses the retained stock and interval-learning banks, the
existing checkpoint, and the retained SINE panel. The runner verifies exact
PCM against the prior records and writes the development points before
scoring comparison inputs.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_shared_mel_context score "$run" \
  --archive "$stock_archive" --sine "$sine_panel"
```
