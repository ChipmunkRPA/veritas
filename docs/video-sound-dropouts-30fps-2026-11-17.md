# Soundtrack dropouts on 29.97 and 30 fps video

Chipmunk Verify's local Full analysis can now look for a brief, near-silent
soundtrack interval while a visible mouth continues moving in eligible
29.97 and 30 fps videos. The source track must have one continuous mono
16 kHz PCM or 48 kHz AAC soundtrack. The picture clock must start at zero
and agree with a regular cadence within 1.5 ms per timestamp. Measurements
and exports use the actual decoded presentation timestamps. The existing
amplitude and mouth-motion checks apply to single shots and separately
within detected shots of a montage.
[Earlier shot-level analysis](video-sound-dropouts-across-cuts-2026-11-17.md)

The ordinary results, PDF and JSON link the measured range to the waveform
and to original frames and mouth crops before, during and after the interval.
The app also identifies the detected shots that could be checked. Quiet
movement, ordinary muting and recording dropouts can produce this signal.
It carries zero weight in the overall AI assessment. The scope requires
34–120 frames per checked shot, up to five shots, and at most 360 video frames
and 15 seconds of presented audio. If a shot contains several qualifying
dropouts, the check reports its longest verified interval.

## Local comparison

The fixed panel selected 32 neutral acted speech recordings from different
CREMA-D speakers absent from the earlier 45-speaker native timing panel.
The acquisition verifies each file against the publisher's pinned Git LFS
length and SHA-256. The native deliveries keep the recorded frame spacing.
The exact-30 deliveries assign 30 fps timestamps to those same pictures.
The comparison verifies that every decoded picture matches the recording.
The audio is aligned to the first presented picture, within the recorded
millisecond clock precision, and trimmed or padded at the end to the picture
extent. The receipt records both adjustments. The 30 fps montages combine
two of those recordings and a 0.7-second title card.
[Corpus and terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt)
· [Earlier native timing panel](video-lip-carrier-transfer-2026-11-16.md)

Each presentation compares paired 240, 360 or 520 ms hard and faded mutes
over active sound. The control deliveries keep the soundtrack as rendered,
reduce the entire track's gain, or reduce only the chosen interval to 10%
of its amplitude. Montages also test a mute that crosses a detected cut.
A correct location needs at least 0.40 temporal intersection over union with
the staged interval. Soundtrack treatments share the same pictures and source
groups. The older near-silence route assessed zero of these presentations.

| Picture presentation | Correct hard mutes | Correct faded mutes | Controls flagged |
| --- | ---: | ---: | ---: |
| Native 29.97 fps, one shot | 30/32 | 30/32 | 0/96 |
| Exact 30 fps, one shot | 30/32 | 30/32 | 0/96 |
| Exact 30 fps, three shots | 11/12 | 11/12 | 0/48 |

The counts were identical for the PCM and AAC deliveries. Every delivery
completed the check, and every reported range reached the required overlap
with a staged mute. The montages matched all planned cut boundaries. Each
title-card shot was listed as outside the measured range. The numbers
describe staged dropouts in short clips of visible speakers. A replaced
voice with an audible soundtrack requires a different check.
[Results and input commitments](evaluation-panels/video-sound-dropouts-30fps-2026-11-17.json.gz)

## App verification

Ten simulated USD 9.99 checkouts completed in the isolated local app.
Each upload created its analysis job only after paid confirmation. Four
edited uploads produced the same source ranges and frames in the ordinary
view, PDF and JSON. The matched unedited controls had no qualifying
finding. All four signed evidence packages replayed against their original
uploads. The browser decoded the three frames, three mouth crops, waveform
and audio excerpt in each finding. The paired switch-off run for each frame
rate returned the same overall AI assessment.
[Local checkout and export receipt](evaluation-panels/video-sound-dropouts-30fps-app-2026-11-17.json)

## Reproduce the checks

From the repository root, with the project's local video assets and FFmpeg
available, run the following. The source recordings and derived media remain
in the local run directory. Numeric results retain the input commitments,
original presentation ticks, edit ranges and observed shot coverage.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
repo=$(pwd)
run=$(mktemp -d /tmp/chipmunk-video-dropouts.XXXXXX)
publisher=https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/1658cd342dff90010aa843eaeebd53610a08b1dc
earlier=docs/evaluation-panels/video-lip-delivery-native-2026-11-16.json.gz
runner=backend/.venv/bin/python
assets="$repo/models/research/video-lip-timing"
curl -fL "$publisher/SentenceFilenames.csv" -o "$run/SentenceFilenames.csv"
curl -fL "$publisher/LICENSE.txt" -o "$run/LICENSE.txt"
"$runner" -B -m scripts.video_silent_mouth_clock_panel acquire \
  "$run/SentenceFilenames.csv" "$run/LICENSE.txt" "$earlier" "$run/sources"
"$runner" -B -m scripts.video_silent_mouth_clock_panel measure "$run/sources" "$assets" "$run/panel"
"$runner" -B -m scripts.video_silent_mouth_clock_panel review "$run/panel/result.json.gz"
npm --prefix frontend run build
(cd backend && PYTHONPATH=. .venv/bin/python -B -m scripts.video_lip_multishot_app \
  "$run/panel" "$assets" "$run/app-native" --browser --clock native-29.97)
```

The app replay also accepts `--clock exact-30` and `--clock montage-30`.
