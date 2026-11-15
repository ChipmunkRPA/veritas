# Reconstructing cleaner speech features before short-edit detection

This trial tests whether learning from paired clean and noisy recordings helps
locate a short synthetic voice edit under competing speech. The local detector
learns a correction to a frozen speech representation. Each assessment uses
one six-second upload. The same timed detector architecture and speech
admission run before and after the change. Work on noisy multi-speaker
localization and separate boundary and segment cues informed the comparison.
[Noise and multi-speaker research](https://arxiv.org/abs/2507.16220) ·
[Boundary and segment research](https://arxiv.org/abs/2609.25822)

## Fit and controls

Forty-eight untouched LibriSpeech prompts from six fitting readers supply
paired clean and loud-mixture features. The fit uses neither synthetic speech
nor edit timestamps for this feature correction. For each frame, a ridge
regression reads the center and two neighboring states at ±20 ms. It predicts
the difference from the loud mixture to its clean counterpart and uses zero
correction as the target on clean inputs. The regression uses fixed
standardization and a penalty of 0.25 times the number of fit frames. The
learned map is the same for every clip. Only the existing frozen encoder, the
existing small localization architecture and this local fit are used.
[Encoder assets and weight terms](audio-wavlm-v13-protocol.md) ·
[Existing localization method](audio-short-edit-speech-layers-2026-11-14.md)

The two detection heads fit one synthetic voice family each and evaluate the
other. Separate genuine calibration readers set both cutoffs before held
scoring. Held mixture backgrounds come from other meeting rooms. Every
staged edit has a matched untouched file, a genuine splice, an ordinary
resampling and an isolated background at the same clock. The projection
evaluation reuses the committed PCM and cached features from the earlier
study. [Source separation and corpus terms](audio-short-edit-detector-training-2026-11-14.md#source-and-fitting-roles)

## Results under competing speech

The corrected representations move closer to the exact clean counterpart on
all 144 held paired deliveries. The median normalized squared error fell
between 27.1% and 28.8% across the four case types. Synthetic edits outranked
all three matched genuine recordings in 19/48 groups, compared with 18/48 for
the prior features. Lower feature error gave little change in detection.

The table uses the wider genuine-calibrated point. Ordinary controls exclude
the separate isolated backgrounds. A useful range covers at least 50% of the
known edit and reaches 0.30 temporal intersection over union. The broader
control comparison treats a useful range in any matched control at the same
edit clock as a collision, even when its endpoints differ from the edit
file's predicted range.

| Held voice family | Check | Useful /24 | Useful with no edit-clock control match | Ordinary flags /72 | Isolated background flags /24 |
| --- | --- | ---: | ---: | ---: | ---: |
| Chatterbox | Prior frozen features | 1 | 1 | 16 | 2 |
| Chatterbox | Corrected features | 2 | 0 | 10 | 2 |
| MeloTTS | Prior frozen features | 1 | 0 | 4 | 2 |
| MeloTTS | Corrected features | 1 | 1 | 0 | 0 |

Both Chatterbox locations from the corrected features occurred at the same times on the
isolated backgrounds. All four useful ranges from the current short speech
review also occurred in matched controls. At the strict point, the two
corrected heads found zero useful edited ranges and flagged zero ordinary or
background files. All compared deliveries had eligible speech proposals.

On the separate 48-recording transfer panel, the fit using Chatterbox changed
from one useful ElevenLabs location and zero ordinary flags to zero useful
locations and three flags. The fit using MeloTTS retained the same useful
ElevenLabs location. It flagged four ordinary files, compared with three
before correction. Both versions of that result had an ordinary control
overlap the edit clock with different endpoints. Neither projected head
usefully located any of the three clone or four Kokoro edits. The original
fit using Chatterbox retains the only transfer location that clears the
matched-control check. The separate panel used retained files from four
other rooms and three generators outside the fit. [Transfer source roles](audio-short-edit-speech-layers-2026-11-14.md#separate-transfer-test)

The reviewer also compares ranking with retrospective budgets of zero, one,
two and four ordinary groups and zero isolated-background flags. On the
separate panel, at two ordinary groups, the strongest prior check found
three useful locations absent from matched controls. The corrected fits
found one and zero, respectively. These budgets describe the held data and
were not used to set the operating points. The projection remains in
evaluation. None of these scores enters customer confidence.

## Independent replay

The public receipts include scalar distances, responses, sample ranges and
commitments. The matching clean and noisy recordings and their features stay
local. The numeric reviewer independently checks the cutoffs, recounts the
matched-clock controls, and reconstructs the equal-alarm comparison without
loading a model.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
panels=docs/evaluation-panels
module=scripts.audio_short_edit_noise_projection
python3 -B -S -m "$module" review "$panels/audio-short-edit-detector-training-2026-11-14.json" \
  "$panels/audio-short-edit-speech-layers-2026-11-14.json" \
  "$panels/audio-short-edit-noise-projection-2026-11-15.json"
python3 -B -S -m "$module" review-transfer "$panels/audio-noise-projection-transfer-2026-11-15.json"
```

Reconstruct the inputs using the [original source preparation](audio-short-edit-detector-training-2026-11-14.md#independent-replay),
the [frozen encoder cache](audio-short-edit-speech-layers-2026-11-14.md#independent-replay)
and the [separate transfer audio replay](audio-remote-neighbor-replication-2026-11-12.md#replay).
A second local CPU fit reproduced the projection, both small heads and all
primary measurements byte for byte. A second native transfer reproduced the
48-file result byte for byte.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
runner=backend/.venv/bin/python
"$runner" -B -m "$module" train "$sources" "$cache" "$speech_asset" "$run/fit"
"$runner" -B -m "$module" transfer "$sources" "$transfer" "$encoder" "$speech_asset" \
  "$run/fit" "$panels/audio-short-edit-speech-layer-transfer-2026-11-14.json" "$run/transfer"
```
