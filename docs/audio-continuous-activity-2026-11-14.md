# Speech coverage when background noise masks a short edit

The overlapping voice review now reports how much speech it detected
inside each three-second window. A window below the 1.500-second speech
minimum is marked separately from a scored but inconclusive window.
The analysis view, PDF and JSON show the same duration and source
times. Speech activity measures the submitted mix and can include
any audible speaker. The independent verifier can reproduce these
measurements from the signed export and original audio.

## Local comparison

The worker ordinarily restarts the recurrent speech-activity check in
each three-second window. We compared that approach with a single pass
over the recording whose measured speech spans were intersected with
the same windows. One comparison uses the continuous result alone.
Another preserves all existing readings and scores additional windows
that pass the same half-window rule. The voice classifier scores the
original, unprocessed samples at its existing calibration. The
overlapping voice review combines adjacent three-second AI-like
readings. It remains separate from the file-level AI score.

The 98 deliveries match the earlier rain and train study. They include
18 partially edited foregrounds from six generators and 12 genuine
foregrounds, each delivered three ways, plus eight isolated background
recordings. The two added backgrounds were matched at −5 dB. Repeated
foregrounds and previously measured sources limit transfer. No new
weights or operating points were introduced. [Source and license
records](audio-focused-window-transfer-2026-11-12.md#frozen-comparison)
· [Model and activity asset terms](commercial-model-audit.md)

| Condition | Role | Windows scored after reset | Windows scored with continuous activity | Newly scored |
| --- | --- | ---: | ---: | ---: |
| Gain only | Partial edits | 96/96 | 96/96 | 0 |
| Gain only | Genuine | 61/61 | 61/61 | 0 |
| Rain | Partial edits | 88/96 | 94/96 | 7 |
| Rain | Genuine | 58/61 | 61/61 | 3 |
| Train | Partial edits | 30/96 | 64/96 | 36 |
| Train | Genuine | 20/61 | 29/61 | 11 |
| Rain and train | Background only | 0/40 | 0/40 | 0 |

The continuous route also removed one previously scored rain window
and four train windows. The additive route preserved all previously
scored windows. Among 36 newly scored train windows from partially
edited files, three leaned AI and 33 were inconclusive. One genuine
train-masked file gained a false range. One edited train-masked file
gained a range with insufficient overlap with the edit. Useful
localization remained **13/18** for gain only, **3/18** for rain and
**0/18** for train in all three comparisons. A useful overlap was
geometrically possible inside an adjacent pair for 15 of the 18 train
edits. Speech admission left one of those reachable after a reset and
ten reachable with the continuous pass. The voice readings still failed
to locate any of them. The continuous approach stays experimental.
[Evaluation receipt](evaluation-panels/audio-continuous-activity-2026-11-14.json)

## Local app verification

Six simulated prepaid runs compared the known rain-masked edit, a
genuine rain mixture and isolated rain with the 48 kHz trial on and
off. No analysis started before payment and the overall decisions
matched. The PDF, API and JSON showed the four measured speech counts
behind the **3.000–4.500 s** finding and its supporting windows. The
signed replay reproduced all four counts from the submitted WAV.
The isolated-rain PDF showed **0.000 of 3.000 seconds** of speech and
the minimum requirement for each of its five skipped windows.
[App replay receipt](evaluation-panels/audio-continuous-activity-app-2026-11-14.json)

## Replay

Reconstruct the inputs as described in the [parent study](audio-focused-window-transfer-2026-11-12.md#replay).
The public receipt contains only counts, directions and window times.
The standard-library recount verifies the selected ranges and outcomes.
The native replay recomputes both speech-activity passes from the
original PCM using the pinned activity weights.

```sh
export PYTHONPATH=backend VERITAS_AUDIO_DETECTOR_MODE=candidate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
receipt=docs/evaluation-panels/audio-continuous-activity-2026-11-14.json
python3 -B -S -m scripts.audio_continuous_activity_transfer review "$receipt"
backend/.venv/bin/python -B -m scripts.audio_continuous_activity_transfer review "$receipt" --native "$parent"
backend/.venv/bin/python -B -m scripts.audio_continuous_activity_transfer run "$run" --native "$parent"
```

To verify the measured coverage in one signed customer export, obtain
the instance public key independently of the ZIP. The reviewer checks
the exact audio samples and recomputes the three-second speech counts.
Only the speech-activity weights are needed.

```sh
backend/.venv/bin/python -B -m scripts.verify_short_speech_coverage \
  "$signed_package" "$original" --public-key "$trusted_key" --long-review \
  --activity-weights models/weights/audio-upgrade/silero_vad.jit
```
