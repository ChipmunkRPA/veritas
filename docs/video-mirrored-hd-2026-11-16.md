# Mirrored footage in HD and portrait video

The mirrored-footage check now covers the opening eight seconds of
supported 720p and 1080p videos in landscape and upright portrait.
It reports both source ranges and four original frame images.
The PDF, JSON and signed evidence package use the same source
timestamps. A reflected repeat can come from ordinary editing.
The check uses no model weights and contributes zero AI confidence.

## Native picture clock

The check accepts regular 23.976, 24, 25, 29.97 and 30 fps
cadence in supported 8-bit 4:2:0 H.264, VP9 and HEVC.
The fractional rates are exactly 24000/1001 and 30000/1001.
It compares measured presentation timestamps with the cadence of
the complete checked prefix. Where the cadence falls between
container ticks, as it does at 29.97 fps in typical WebM, it allows
the relative error from rounding individual timestamps to a tick.
When the nominal period is a whole number of ticks, it requires
exact spacing. The original ticks still label each range and its
frames. A gap or inferred timestamp can prevent a result.
[Matroska timestamp units](https://www.matroska.org/technical/notes.html)
· [FFmpeg frame timestamps](https://ffmpeg.org/doxygen/trunk/structAVFrame.html)

Area averaging reduces the decoded native RGB pictures to the
40 × 30 luminance grid used for the reflection comparison.
The check measures per-frame visual similarity and the direction
of motion between frames. Its motion floor adjusts for nominal
frame rate. It reuses the existing bounded native decode and stores
the full-resolution original endpoint pictures. The two ranges can
have one tick of difference in duration and each keeps its own
measured timestamps. The previous 640 × 360 profile keeps its
existing cadence and comparison. [First mirrored-footage evaluation](video-mirrored-sequence-2026-11-16.md)

## Held comparison

Nine new camera recordings supplied 144 deliveries across six
profiles. Six recordings came from distinct Commons contributors
and three from the Microsoft VCD2 camera collection. Some source
recordings appear in more than one profile. Portrait deliveries
include an upright portrait original and center crops of landscapes.
The ledger includes source, license, digest, and frame-range
commitments. [Held delivery ledger](evaluation-panels/video-mirrored-hd-2026-11-16.json.gz)
· [VCD2 license](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA)

Each profile has three matched original groups. Each group supplies
a short mirrored reuse, a longer reuse in the opposite order, and
a second encode of the short reuse. Five controls per group cover
the original, an in-place flip, a still hold using a mirrored frame,
a same-orientation repeat, and a second encode of the original.
Both reported ranges must reach at least one-half intersection
over union with their authored range. The earlier mirrored check
leaves all these HD profiles unassessed.

| Delivery | Mirror locations | Locations after a second encode | Control alerts |
| --- | ---: | ---: | ---: |
| 720p H.264 MP4 at 25 fps | 5/6 | 2/3 | 0/15 |
| 1080p H.264 MP4 at 29.97 fps | 4/6 | 2/3 | 0/15 |
| 720p VP9 WebM at 29.97 fps | 4/6 | 1/3 | 0/15 |
| 1080p VP9 WebM at 24 fps | 6/6 | 3/3 | 0/15 |
| 1080 × 1920 HEVC MOV at 30 fps | 6/6 | 3/3 | 0/15 |
| 720 × 1280 H.264 MP4 at 23.976 fps | 6/6 | 3/3 | 0/15 |
| Total | 31/36 | 14/18 | 0/90 |

All test deliveries were assessed. Both ranges were exact in 30
single-encode edits and in 14 twice-encoded edits. These
counts describe the listed source groups and edits. Variable-rate
uploads, rotation-tagged clips, HDR footage, combined transforms,
and footage beyond the opening prefix remain outside this pilot.

## Local app verification

Separate FFprobe and FFmpeg reads verified the presentation ticks and
RGB hashes of all 180 reported endpoints. Sixteen simulated purchases
queued zero jobs before payment. The ordinary view, API, PDF, JSON,
and signed package agreed on four original endpoint frames and both
ranges for each of four upload profiles. Browser seeking reached
the measured source position for H.264 MP4, VP9 WebM and upright
portrait H.264. The HEVC replay verified browser frame images and
the signed source ranges. The overall assessments agreed on paired
runs with the feature enabled and disabled. [Local app replay](evaluation-panels/video-mirrored-hd-app-2026-11-16.json)

## Replay

The local development option is
`VERITAS_VIDEO_REFLECTED_SEQUENCE_REVIEW=true`. Restore the ledger's
licensed sources locally, check each SHA-256, and add the absolute
path to each row. Use one common source array, deduplicated by its
source ID. The builder produces a before-scoring plan for each
profile. Compare and verify the output with a separate FFprobe
clock read and an FFmpeg frame-hash decode.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
"$runner" -B -m scripts.video_reflected_hd_panel build "$run/$profile" "$sources" "$profile"
"$runner" -B -m scripts.video_reflected_hd_panel score "$run/$profile" "$run/$profile.json"
python3 -B -S -m scripts.review_video_reflected_hd_panel "$run/$profile.json" --media "$run"
```

The portable counter also reads the compressed public ledger
directly. A native startup of the browser replay uses a JSON file
with the six rebuilt profile results in a `profiles` array.

```sh
npm --prefix frontend run build
"$runner" -B -m scripts.video_reflected_sequence_app "$run" "$combined_result" "$local_app_run" --hd --browser
```
