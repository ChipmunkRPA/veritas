# Locate mirrored footage within one video

Chipmunk Verify shows two source ranges where the same motion appears
in opposite left-to-right orientations. Each range comes with its first
and last original frames and playback links. The API, PDF, JSON and
signed evidence package carry the same source times and frames.
Ordinary edits can reuse mirrored footage. The finding leaves the
order of creation undetermined and carries zero AI-confidence weight.

## Comparison

The check compares a 40 × 30 luminance grid from each decoded picture
with the horizontally reversed grid from another picture. A qualifying
frame pair must be sufficiently similar after reflection and distinct
in its original orientation. Both source spans must show the same
direction and timing of picture changes for at least half a second,
with at least eight supporting changes. The ranges cannot overlap.
The review runs on the same decoded pixels and presentation clock as
the exact, similar, reversed and retimed footage checks. It uses no
model weights. Frame-to-frame similarity with temporal alignment is an
established approach in video copy localization. [Video copy localization research](https://arxiv.org/abs/2211.13090)

The fixed comparison ran on eleven newly sourced camera recordings.
Seven came from different Wikimedia Commons contributors, and four
came from distinct VCD2 camera types outside the project's recorded
VCD2 test rosters. For each recording, the two edited deliveries
replace an earlier or later portion with a reflected copy of the
other portion. The short copy covers 17–21 frames. The longer copy
covers 26–32. Ordinary controls include the unchanged clip, an
in-place flip, a mirrored still-frame hold, and a recompressed
unchanged clip. Same-speed and reversed copies are separate controls.
The ledgers include the source credits, license links, delivery hashes,
and the frame ranges committed before scoring. [Commons panel](evaluation-panels/video-mirrored-sequence-commons-2026-11-16.json.gz) · [VCD2 panel](evaluation-panels/video-mirrored-sequence-vcd2-2026-11-16.json.gz) · [VCD2 data terms](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA)

## Local results

A location counts when both proposed ranges overlap their authored
ranges at intersection over union of at least 0.50. The camera
recordings supply paired encodes and edits. Counts refer to deliveries.

| Delivery | Existing repeat checks located both ranges | With the mirrored check | Both ranges exact |
| --- | ---: | ---: | ---: |
| H.264 MP4, one encode | 0/22 | 18/22 | 16/22 |
| VP9 WebM, one encode | 0/22 | 20/22 | 17/22 |
| H.264 MP4, second encode | 0/11 | 7/11 | 5/11 |
| VP9 WebM, second encode | 0/11 | 9/11 | 7/11 |

The check added zero flags on 88 ordinary controls and 44 other
repeat controls. The existing checks retained all 40 flags they had
produced on the latter group. The four VCD2 sources accounted for
four of the six single-encode misses. Separate FFprobe timestamps and
FFmpeg RGB hashes verified all 216 displayed frame endpoints in the
held comparison. The check is scoped to the first 225 upright
640 × 360 pictures on a verified 25 fps clock in 8-bit 4:2:0 H.264 or VP9.
Other sizes, frame rates, combined transforms and later ranges
remain outside this pilot.

## Local app verification

Eight simulated USD 9.99 purchases exercised the worker, with zero
jobs queued before payment. With the check enabled, the test found
**1.520–2.160 s** and **5.960–6.600 s** in both MP4 and WebM. The browser
loaded four original frame images and sought to 5.960 s. The signed
packages verified the original frames and source timestamps. The API,
PDF and JSON agreed. The ordinary and in-place flip controls stayed
clear, and the paired runs with the check disabled produced the same
overall assessment. The replay isolates this model-free check from
the learned detectors. [Application replay](evaluation-panels/video-mirrored-sequence-app-2026-11-16.json)

## Replay

The option `VERITAS_VIDEO_REFLECTED_SEQUENCE_REVIEW=true` enables the
check in development. For a native replay, restore each source URL
from the corresponding ledger, verify its SHA-256, and add its local
`path` to a copy of the `sources` array. Keep the original and staged
videos in the local work directory.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
"$runner" -B -m scripts.video_reflected_sequence_panel build "$run" "$sources"
"$runner" -B -m scripts.video_reflected_sequence_panel score "$run" "$run/result.json"
python3 -B -S -m scripts.review_video_reflected_sequence_panel "$run/result.json" --media "$run"
npm --prefix frontend run build
"$runner" -B -m scripts.video_reflected_sequence_app "$run" "$run/result.json" "$local_app_run" --browser
```

Use the last command with the Commons panel for the prepaid browser and
report replay. The numeric review also runs directly on either published
compressed ledger using only the Python standard library if `--media`
is omitted.
