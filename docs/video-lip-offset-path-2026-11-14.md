# Brief lip-sync changes within a continuous shot

This single-upload experiment compared one constant audio/video lag with
one short change in lag. It learned eight regularized shared directions
from matching speech and mouth observations. It compared both timing
hypotheses on the same measured ticks. Both flanks had to support the
common lag, and the middle had to support the temporary lag over the best
constant fit to all three parts. An observed picture cut prevented the
candidate from joining support across it. The experiment used the
existing MIT-licensed face weights and no external timing model.
[Face weight terms](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)
· [Shot-level temporal alignment research](https://arxiv.org/abs/1808.06250)
· [Timing and articulation research](https://arxiv.org/abs/2610.09223)

## Data and comparison

The study uses 16 credited conversational recordings with commercially
permitted licenses from the earlier local intake. Seven distinct speakers
supply fit clips, four supply the operating point and five supply held
results. Two separate 6.4-second excerpts per source undergo the same
25 fps H.264 and 16 kHz PCM preparation. The fit used 1,199 speech ticks
from 13 qualifying excerpts. Different parts of the same recording stay
in one source split. The five held recordings had earlier project use.
Their new excerpts start at 52 and 79 seconds. The receipt preserves
every source file, license, excerpt offset, encoded excerpt hash and
decoded soundtrack commitment.
[Earlier source credits](evaluation-panels/video-mouth-hd-hold-2026-11-12.json)
· [Separate held source credits and URLs](evaluation-panels/video-local-alignment-2026-11-14.json)

The known local timing shifts range from 200 to 280 ms. Edited intervals
last 320, 480, 640 or 880 ms. Controls include unchanged speech, local
gain, filtering, noise, both whole-track shifts, synchronized picture and
sound changes, dubbing with another person's speech, and rendered cuts
between two synchronized recorded scenes.
The rendered cuts used new encoded pictures and PCM. The synchronized
controls reused measured picture observations with the corresponding
sound shift. The first preparation stopped during fitting when seek and
resampling produced four fewer samples than planned. The completed run
pads or trims to a fixed output sample count and verifies its native
picture and audio clocks. It retained all planned source roles.

The candidate point is the strongest ordinary score from the point
speakers plus 0.02, with a floor. It froze at 0.91222835. Correct
localization needs at least 0.40 overlap by temporal IoU, measured on
the soundtrack clock, and the signed lag within 80 ms. Both candidate
and existing window positions are mapped from picture time to sound time
for this comparison. The candidate checks for at least four useful
additions across three people, and at most two added ordinary alerts and
two added wrong locations. Whole-track, synchronized, dubbed and cut
controls admit zero additional timing alerts.

| Held outcome | Existing check | Temporary-lag comparison |
| --- | ---: | ---: |
| Staged edits with a measurement | 6/20 | 5/20 |
| Ordinary controls with a measurement | 32/100 | 36/100 |
| Correct signed locations | 0/20 | 0/20 |
| Ordinary controls flagged | 0/100 | 0/100 |

Both excerpts from one speaker lacked a measurable mouth and remain in
all relevant denominators. The strongest held proposal stayed below the
frozen point. No strongest edit proposal had useful geometry and lag
even before applying that point. The method stays experimental. This test
addresses localized timing shifts. Synchronized redraws, where mouth shape
and speech content may disagree without an obvious shift, require a separate
measurement. [Per-file timing and control receipt](evaluation-panels/video-lip-offset-path-2026-11-14.json)

## Comparison windows in the app

When the current timing check records a flagged window, new evidence sets
can show the closest measured window on each side. Each comparison must
use the same face track and measurement run. Its recorded interval must
stay entirely before or after the flagged window and within one second of
it. The ordinary browser view and Standard PDF link the two comparisons
back to the flagged window. JSON includes the explicit before or after
role, source timestamps, representative frames, mouth crops and audio.
The Standard PDF shows each linked window's waveform, lag plot and
representative frames. The complete measured input list and asset hashes
are retained in JSON and the evidence ZIP. The comparisons use the
existing timing readings.

The report regression uses source-bound measurements from an encoded
controlled video and commits a flagged interval for the presentation test.
It verifies three disclosed frames per window, exact audio and crop bytes,
the signed Standard PDF, JSON, and rejection of a tampered comparison role.
The frontend test checks the forward and return links. Existing evidence
sets retain the selection policy recorded at creation.

## Replay

The standard-library reader recounts the measured and unassessed files,
ordinary alerts, localization and frozen decision. The native runner
replays the source bytes and frame clocks with the pinned sources. Raw
media and trained representations stay local. Two local runs with the
same encoder and source files produced identical receipt bytes.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
receipt=docs/evaluation-panels/video-lip-offset-path-2026-11-14.json
python3 -B -S -m scripts.video_lip_offset_path_panel review "$receipt"
backend/.venv/bin/python -B -m scripts.video_lip_offset_path_panel run "$run" \
  --cache "$earlier_sources" --held "$held_sources"
cmp "$receipt" "$run/receipt.json"
```

To verify the report and source binding from `backend`, run
`python -m pytest -q tests/test_video_av_evidence.py -k nonoverlapping_source_context`.
