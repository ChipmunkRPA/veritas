# Lip timing at 30 and 29.97 frames per second

The local lip-timing review currently admits short continuous clips in a
slower frame-rate band. To test wider coverage, we kept the application's
existing aperture references, timing search and operating point fixed.
Correct localization requires both the right timing direction (within 80 ms
of the introduced shift) and at least 0.40 temporal overlap by intersection
over union. An ordinary flag and a flag on a uniform whole-clip delay both
count against expanding coverage. A localized timing change can come from
dubbing or editing and does not identify who or what produced it.
[Existing scoring and source evidence](video-lip-viseme-transfer-2026-11-15.md)

## Isolate the picture clock

On 16 previously held Lombard GRID recordings from eight speakers, we
reused the recorded mouth apertures and independently frozen soundtrack
events. We replayed the original timestamps and scheduled the existing
picture measurements on exact 30/1 and 30000/1001 grids. The hold arm uses
the latest picture available at each tick. The nearest arm uses the closest
original picture. Both can repeat measurements. They create no intermediate
frame. The soundtrack and its known edit boundaries stay fixed. This
measurement replay isolates cadence and sampling effects.

The 30 and 29.97 results have identical flag and useful-location totals.
Unresolved case counts can differ by a few files. Each soundtrack carrier
contains 32 local shifts, 48 ordinary originals/filters/gain controls and
32 constant-delay controls drawn from the same 16 recordings.

| Picture schedule | PCM useful local shifts | PCM ordinary flags | PCM constant-delay flags | AAC useful local shifts | AAC ordinary flags | AAC constant-delay flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original picture timestamps | 4 | 0 | 3 | 3 | 0 | 1 |
| Hold the last picture on either new clock | 3 | 0 | 0 | 3 | 0 | 0 |
| Use the nearest picture on either new clock | 4 | 0 | 1 | 3 | 0 | 2 |

Holding the last picture removes the observed constant-delay flags and
loses one correct PCM location. Using the nearest picture retains all
seven correctly located carrier deliveries and adds one constant-delay
AAC flag. A correct result on both carrier versions of one staged edit
counts as two deliveries from the same underlying edit.

## Separately recorded 29.97 fps speech

We selected one recording each from 24 CREMA-D actors by duration,
continuity and deterministic source order. These actors are separate
from the Lombard GRID speakers used for the frozen timing references.
We verified the original file, all 1,824 decoded picture timestamps and
the audited mono PCM digest for each recording. The Flash sources have
only millisecond clock precision. The audit anchors decoded PCM to the
recorded start while keeping the picture timestamps as delivered. The
original audio frame starts deviate from the continuous sample clock by
less than 1 ms across the selected clips. These are single-shot acted
lines and staged timing edits with known bounds.
[CREMA-D recordings and terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/master/README.md#license)

We placed one 700 ms interval at the center of each soundtrack and
generated an early and late version with a 200 ms local displacement.
The comparison also includes the untouched original, filtered and
gain-adjusted versions, and two constant 200 ms delays. The mouth
measurements use the original pictures in every comparison.

| Native speech test | Cases | Flags | Correct localizations | Unresolved |
| --- | ---: | ---: | ---: | ---: |
| Staged local shift | 48 | 2 | 2 | 18 |
| Original, filter and gain controls | 72 | 0 | — | 35 |
| Constant-delay controls | 48 | 1 | — | 25 |

The two correct locations came from separate actors. The frozen rule
missed or could not resolve the other 46 shifts. The wider cadence stays
in evaluation. The app's existing 29.97 fps face-change review is a
separate check. This study does not validate generated mouths or native
30 fps footage from a consumer camera or encoder.
[Native speech aggregate](evaluation-panels/video-lip-native-clock-2026-11-15.json)

## Replay

The picture-clock comparison can be rerun without downloading media or
loading models. Python 3.12 and 3.14 reproduced the results with only the
standard library. The receipt records unresolved cases, paired gains and
losses, and source-clock witnesses.

```sh
export PYTHONPATH=backend
python3 -B -S -m scripts.video_lip_frame_clock_transfer --verify \
  docs/evaluation-panels/video-lip-frame-clock-2026-11-15.json.gz
```

Use the pinned [native source audit](video-native-source-qualification-v92.md)
and [timing assets](video-lip-viseme-transfer-2026-11-15.md#independent-replay)
to rerun the separate picture and speech inference. The command writes a
local receipt with source hashes and per-case results. The published
receipt contains aggregate counts and its digest. A second local native
run reproduced the detailed receipt byte for byte.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
assets=models/research/video-lip-timing
backend/.venv/bin/python -B -m scripts.video_lip_native_clock_transfer \
  "$audit_dir" "$publisher_media_dir" "$assets/checkpoint" \
  "$assets/face_landmarker.task" "$assets/runtime" "$run/native-clock.json"
```
