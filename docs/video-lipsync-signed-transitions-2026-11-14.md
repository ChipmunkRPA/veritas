# Signed mouth and speech transitions for short lip-sync edits

Comparing the direction of mouth opening and closing with rising and
falling speech energy measured more short edits than the current timing
check. It correctly located zero of 66 planned edits at its frozen
operating point. Requiring agreement with an aperture-and-level
measurement still flagged nine controls. The new methods stay in
evaluation.

## Comparison

The ordinary timing check compares one second of visual aperture with
speech energy. This experiment asks whether the direction of change in
each signal carries timing information that their levels miss. The two
measurements use the same verified 25 Hz picture and 16 kHz sound grid.
One uses mouth aperture and speech RMS. The other uses signed
first differences of visual aperture and compressed speech RMS. The
signed method requires distinct observed pictures for successive inputs.
For a candidate 480, 640, or 880 ms interval, both 480 ms flanks must
prefer a common lag, and the middle must prefer a different lag. The
agreement arm requires both measurements to support the same interval and
lag pair. Recent research distinguishes coarse synchronization from
agreement between specific speech sounds and visible mouth shapes. This
experiment measures synchronization on an existing geometry asset.
[Timing and articulation study](https://arxiv.org/abs/2610.09223)
· [Lip-focused consistency study](https://arxiv.org/abs/2610.30872)
· [Face asset license](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

Each method scored its strongest candidate before applying its point.
The points were frozen from the highest matched ordinary development score
plus 0.03, with a 0.10 floor. The three points were 1.408968 for
aperture and level, 0.880277 for signed transitions, and 0.821915 for
agreement. These are correlation-difference units.

The roster reused 15 actor files and 11 conversational recordings from
the earlier timing study, with four actors reserved for development. An
additional excerpt starts four seconds after each conversation's first
excerpt. These are new excerpts from the same speakers. Five held actor
files were too short and remain in the denominator. Each excerpt has two
temporary sound shifts (±200 ms or ±280 ms), an unchanged copy, local gain,
low-pass and noise controls, both constant sound offsets, and two
synchronized picture-and-sound controls. The synchronized controls
reassign verified mouth observations without a new encode. All other
picture inputs are unchanged. The earlier study documents the exact
publisher files, offsets, and applicable licenses.
[Source commitments and clock preparation](video-lipsync-window-schedule-2026-11-14.md#frozen-comparison)

## Results

Useful localization means at least 0.40 intersection over union and signed
lag within 80 ms of the known shift. Counts include all planned held cases.

| Method | Edits measured /66 | Controls measured /264 | Useful edits /66 | Wrong interval or lag /66 | Control flags /264 | Constant or synchronized flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current timing | 33 | 128 | 0 | 0 | 0 | 0 |
| Short aperture and level | 40 | 145 | 0 | 1 | 19 | 8 |
| Signed transitions | 40 | 145 | 0 | 6 | 32 | 16 |
| Agreement | 40 | 145 | 0 | 1 | 9 | 5 |
| Current or agreement | 40 | 150 | 0 | 1 | 9 | 5 |

For example, the agreement arm flagged 1.240–2.120 s with a +160 ms
lag change on conversational group 10 when the picture grid and speech
had been moved together. Agreement between the two measurements was
therefore insufficient to rule out this synchronized control.
For agreement, five of the strongest unfiltered edit proposals had
the right interval and lag. All five scored below the frozen point and
below at least one already flagged control. Signed transitions had ten
such proposals. The measurements do not support a product change.

## Replay

The [receipt](evaluation-panels/video-lipsync-signed-transitions-2026-11-14.json)
contains the selected correlations, declared clocks, sample ranges and
source frame hashes. The plain
Python reviewer replays the operating points, disposition, source-time
relationships and error counts. Native replay decodes each qualified
excerpt, measures the two signals again and reconstructs the exact
before, during and after timestamps for flagged proposals. Two native
runs produced identical receipt bytes. The reviewer rejected separate
changes to a published point, sample position, picture timestamp and score.

For the actor cache, save the [actor intake receipt](evaluation-panels/video-lipsync-native-actor-intake-2026-11-11.json)
as `intake-receipt.json`. Obtain each file from the [pinned publisher tree](https://github.com/CheyneyComputerScience/CREMA-D/tree/1658cd342dff90010aa843eaeebd53610a08b1dc/VideoFlash)
using its `source_path` and store it under the receipt's `file` value. For
the conversation cache, obtain the original from each Commons file page
in the [conversation plan](evaluation-panels/video-mouth-hd-hold-2026-11-12.json)
and name it `<name>.webm`. The runner verifies every publisher SHA-256
before decoding.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
receipt=docs/evaluation-panels/video-lipsync-signed-transitions-2026-11-14.json
python3 -B -S -m scripts.review_video_lipsync_signed_transitions "$receipt"
for phase in freeze development held; do
  backend/.venv/bin/python -B -m scripts.video_lipsync_signed_transitions "$phase" \
    "$actor_cache" "$conversation_cache" "$new_run" || exit 1
done
cmp "$receipt" "$new_run/receipt.json"
```
