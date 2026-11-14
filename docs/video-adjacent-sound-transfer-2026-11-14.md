# Adjacent repeated speech in short videos

The new check recovered eight useful pairs missed by the prior repeated-audio
method on the short timing panel. Keeping both methods found 15 of 44
planned shifts versus seven using the prior method alone. The four-recording
AAC transfer added no clean locations and found one edit under competing
speech. The check remains an offline evaluation.

## Method and sources

A local soundtrack shift can leave two adjacent copies of the same sound.
The new comparison searches within 600 ms excerpts for candidate lags of
120–360 ms. It requires 60 ms waveform windows at the same lag in two
speech bands and retains nonoverlapping pairs of at least 120 ms. Their
distance can exceed their duration by up to 40 ms. Speech activity and the
same moving-mouth comparison are applied to both audio methods. At least
three distinct source pictures per range must support the visual result.
The test uses the existing permitted face-geometry weights. It implements
its own signal processing. Onset and temporal consistency measure different
properties from phoneme-level alignment. [Face asset license](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)
· [Recent timing and articulation study](https://arxiv.org/abs/2610.09223)

The first panel reused the 26 planned acted and conversational recordings
from the window-schedule study. It retained the two signed 200 or 280 ms
soundtrack edits, original and local gain, filtering and noise controls,
and two whole-track offsets. It added shifted picture-and-sound controls
in both directions and a repeated quiet bed. In this first comparison
the synchronized controls shift the recorded native frame choices on the
same measurement grid without encoding a new video. Five held actor
recordings were too short. They remain in the planned denominator.
[Original intake and license terms](video-lipsync-window-schedule-2026-11-14.md#frozen-comparison)

Useful means that both reported ranges overlap the known repeated portions
at 0.50 temporal IoU or higher. The paired ranges do not identify which
copy was changed, the full edit boundaries, or whether AI was involved.
The earlier one-second app timing test used a different success criterion
requiring a signed full-edit location. It found none of the 44 edits in
that test. These results evaluate repeat witnesses.

## Results

| Held panel | Prior repeat and mouth | Adjacent repeat and mouth | Combined useful | Controls flagged, prior / adjacent |
| --- | ---: | ---: | ---: | ---: |
| Acted | 2/22 | 3/22 | 4/22 | 1 / 0 of 99 |
| Conversation | 5/22 | 8/22 | 11/22 | 3 / 0 of 99 |
| Total | 7/44 | 11/44 | 15/44 | 4 / 0 of 198 |

The proposed method found eight added pairs across seven sources, missed
four pairs found by the prior method, and produced no ordinary or misplaced
flags. The prior method alone accounted for four ordinary flags on repeated
beds. The mouth measurement removed all 18 synchronized flags from the
adjacent audio-only result. As an example, one conversational clip returned
**1.200–1.480 s** and **1.480–1.760 s** on the decoded audio clock, exactly
matching the authored repeat. It used frame pairs 30/37 through 36/43.
[First-panel receipt](evaluation-panels/video-adjacent-sound-transfer-2026-11-14.json)

The transfer ran four pinned public recordings, one also in the first
panel at a different offset. The other three speakers are separate from the
first panel. The media had been used elsewhere in the project. Each
recording supplied two edit directions in each of two noise conditions
and two carriers, for eight edited deliveries per recording, plus controls.
The AAC tracks were encoded
at 48 kbps in MP4 and decoded back onto the verified picture clock. For
matching picture-and-sound controls, the runner retimed, encoded and decoded
the complete picture stream. The competing speech at −5 dB came from a
different recording. It was mixed after the foreground edit and scaled
as needed to avoid clipping. The carriers share source and edit identities.
[Transfer sources and licenses](video-lipsync-repeat-fresh-actors-2026-11-13.md#fixed-method-and-sources)
· [Transfer receipt](evaluation-panels/video-adjacent-sound-delivery-2026-11-14.json)

| Transfer | Prior useful | Adjacent useful | Combined useful | Controls flagged, prior / adjacent |
| --- | ---: | ---: | ---: | ---: |
| PCM, clean | 4/8 | 1/8 | 4/8 | 1 / 0 of 36 |
| AAC, clean | 4/8 | 2/8 | 4/8 | 1 / 0 of 36 |
| PCM, competing voice | 0/8 | 1/8 | 1/8 | 0 / 0 of 12 |
| AAC, competing voice | 0/8 | 1/8 | 1/8 | 0 / 0 of 12 |

The same noisy edit was found in both carriers at **0.600–0.880 s**
and **0.880–1.160 s** on the decoded soundtrack clock. The new method
added zero clean AAC locations and missed the two-recording advancement
requirement. The effect under competing speech is one successful case.
The current app has no new lip-sync finding from this experiment.

## Replay

The published receipts contain scalar outcomes, frame positions and hashes.
A reader using only standard Python checks the paired results and recounts
the decisions. Fresh local executions of both native runners reproduced
both receipts byte for byte.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_video_adjacent_sound \
  docs/evaluation-panels/video-adjacent-sound-transfer-2026-11-14.json
PYTHONPATH=backend python3 -B -S -m scripts.review_video_adjacent_sound_delivery \
  docs/evaluation-panels/video-adjacent-sound-delivery-2026-11-14.json
```

For native replay, prepare the original actor and conversation folders
from the [window-schedule intake](video-lipsync-window-schedule-2026-11-14.md#replay).
For the delivery test, fetch the originals listed in the [public source plan](evaluation-panels/video-lipsync-repeat-public-speech-sources-2026-11-13.json)
as `<id>.webm`. Pass a new output path to each command.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze score; do
  backend/.venv/bin/python -B -m scripts.video_adjacent_sound_transfer \
    "$phase" "$actor_cache" "$conversation_cache" "$output" || exit 1
  backend/.venv/bin/python -B -m scripts.video_adjacent_sound_delivery \
    "$phase" "$public_cache" "$delivery_output" || exit 1
done
```
