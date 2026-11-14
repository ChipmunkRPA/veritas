# Denser timing windows on short videos

Three changes to the existing video timing schedule increased the number
of clips with measured windows. None correctly located a temporary audio
shift on the held panel. The two denser schedules also reported ordinary
processing and constant audio delays as local changes. The comparison
stays in evaluation.

## Frozen comparison

The current check compares one second of mouth motion with audio energy
every half second. It searches equally in both directions up to 600 ms.
The comparison tested starts every 200 ms, a search limited to 320 ms,
and both changes together. It kept the same speech gate, measured mouth
variation, speaker attribution and change rules. Each arm required its
full search margin on both sides. The narrower search used the same
attribution stage as the current check and restricted only the lag
comparisons. All four arms used the same verified native mouth inputs
and one directly observed face per frame.
[Existing timing implementation](../backend/app/detectors/video/xcorr.py)
· [Earlier short-window comparison](video-lipsync-local-delay-transfer-2026-11-11.md)

Four recorded actors were development sources. The held roster comprised
eleven other CREMA-D actors and eleven Wikitongues speakers. The actor
recordings retain their original pictures. The conversational recordings
use a fixed 3.52-second excerpt starting 12 seconds after the prior
study's selected offset, outside its eight-second hold trial. These are
new timing deliveries from previously acquired recordings. Five held
actor clips were too short to support the original full lag search on
any 1-second window and stayed in the planned denominator. The other
17 held recordings supplied eight matched deliveries each.
[CREMA-D terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt)
· [Conversation source commitments and per-file licenses](evaluation-panels/video-mouth-hd-hold-2026-11-12.json)
· [Existing face weight terms](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

Two cases per recording move audio earlier or later by 200 or 280 ms
within a 640 or 880 ms interval. Six controls preserve the original
audio or apply local gain, filtering, noise, or one constant
shift to the whole clip in either direction. Useful localization
requires at least 0.40 temporal intersection over union and the
signed lag within 80 ms of the known change. The frozen review requires
four added useful edits across at least three source groups with both
banks represented, at most one added control flag, no new constant-shift
flags, preservation of all current correct locations, and 75% edit
measurement coverage in each bank.

## Held results

| Schedule | Edits measured /44 | Controls measured /132 | Edits with four consecutive windows /44 | Useful signed edit locations /44 | Control flags /132 | Constant-shift flags /44 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current | 19 | 60 | 0 | 0 | 0 | 0 |
| Denser only | 28 | 82 | 18 | 0 | 18 | 5 |
| Narrower only | 26 | 85 | 12 | 0 | 0 | 0 |
| Denser and narrower | 28 | 81 | 21 | 0 | 27 | 11 |

The denser-only arm flagged four edited files at an incorrect range or
with an incorrect lag. The combined arm did so on ten edited files.
Four adjacent windows can overlap heavily, so the extra windows do
not establish four independent observations. The reported results are
from one-second intensity correlation and do not measure the accuracy of
a phoneme-level articulation check. Recent methods directly compare
mouth shape with speech sounds, including under coarse synchronization.
That distinction matters for generators that keep the mouth on time.
[Timing and phoneme-level articulation study](https://arxiv.org/abs/2610.09223)
· [Lip-focused content consistency study](https://arxiv.org/abs/2610.30872)

## Replay

The [window ledger](evaluation-panels/video-lipsync-window-schedule-2026-11-14.json)
contains source commitments, hashes of constructed PCM, measured window
starts, and selected time ranges. It contains no audio, imagery or learned
features. The standard-library reader recomputes the coverage, exact
localization rule and selection decision. The native runner verifies the
26 source hashes, recreates source cadences and remeasures verified mouth
crops. Use the archived actor folder with its original intake receipt and
the folder containing the eleven publisher WebMs.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
receipt=docs/evaluation-panels/video-lipsync-window-schedule-2026-11-14.json
python3 -B -S -m scripts.review_video_lipsync_window_schedule "$receipt"
for phase in freeze score; do
  backend/.venv/bin/python -B -m scripts.video_lipsync_window_schedule "$phase" \
    "$actor_cache" "$conversation_cache" "$new_run" || exit 1
done
cmp "$receipt" "$new_run/receipt.json"
```

Identical receipt bytes require the same media runtime and detector source
revision. The reader works separately from that runtime.
