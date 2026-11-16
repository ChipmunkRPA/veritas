# Reused footage after strong recompression

Chipmunk Verify can locate two ranges of reused footage when one plays
at twice the speed of the other, even after a stronger second encode.
The finding gives both ranges on the submitted video's clock and shows
each range's start and end frames. Its PDF and JSON use the same locations.
Ordinary editing can produce this evidence. The finding carries zero
weight in the overall AI assessment.

## Method

The original comparison requires a close match in picture detail and
in the direction of motion across two native frame steps in the slower
range and one in the faster range. When it finds no match, a second
pass averages four adjacent cells in the existing luminance grid and
compares the resulting broader picture detail. Both passes require
disjoint time ranges, repeated motion support, and at least half a
second of matching time in the shorter range. The coarser pass keeps
the existing priority for exact, equal-speed and reverse comparisons.
It uses no new weights. [Previous replay-speed work](video-retimed-sequence-2026-11-16.md)
· [Research on temporal copy detection](https://arxiv.org/html/2501.11171v1)

Four archived recordings supplied development cases for comparing
grid averages and spatial smoothing. Twelve separate VCD2 recordings
with distinct camera types supplied the held test. These were distinct
from the earlier retiming studies and had been used as camera controls
elsewhere in the project. The final pass used four authored variants
per source, with both orientations of fast and slow reuse. Each variant
had a single H.264 CRF 25 encode and a second delivery from that encode
at CRF 33. The source hashes, labels and comparison parameters were
recorded before held measurement. [Camera data terms](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA)

## Local comparison

A correct match requires the two-to-one ratio and at least 50% overlap
by intersection over union with each labeled range. Counts refer to
deliveries. The 48 variants per encode come from 12 source recordings.

| Outcome | Current check | With the broader picture comparison |
| --- | ---: | ---: |
| Correct ranges after one encode | 37/48 | 38/48 |
| Correct ranges after a second encode | 2/48 | 29/48 |
| Ordinary controls flagged | 0/60 | 0/60 |
| Existing equal-speed or reverse results retained | 15/24 | 15/24 |

The 27 additional second-encode matches cover eight source recordings.
Twelve second-encode deliveries return both labeled endpoints exactly.
Nineteen of the 48 recompressed variants still lack a correct result.
The ordinary controls include untouched recordings, a recompressed
original, an in-place speed change, a jump cut and a held picture. The
test covers staged frame skipping and duplication. Other speed ratios,
interpolated movement, and framing changes need separate evaluation.
FFprobe and FFmpeg separately reproduced all 268 reported endpoint
timestamps and RGB hashes across the 67 matched deliveries.
[Per-delivery comparison](evaluation-panels/video-retimed-compression-2026-11-16.json.gz)

## Application scope and replay

The existing `VERITAS_VIDEO_RETIMED_SEQUENCE_REVIEW` development setting
enables both passes. The comparison uses the first 225 pictures in an
upright 640 × 360 H.264 MP4 with a regular native 25 fps clock and 8-bit
4:2:0 chroma. Each packet records which pass ran and identifies all
measured frames in both ranges. With the original provided, the signed
verifier reruns the two comparisons and checks the displayed frame
pixels and timestamps.

Six simulated USD 9.99 purchases exercised the regular app worker,
with the repeated-footage check isolated from learned detectors. No
analysis queued before payment. Both second-encode examples produced
matching ranges in the browser, API, JSON and PDF. The signed export
verified both against their submitted videos. The first example showed
**1.040–2.640 s** and **5.280–6.080 s** with four original
frames, and the browser sought to **5.280 s**. The earlier direct
match kept its prior result. Both ordinary controls showed no retiming
finding. The tested overall assessment agreed with the check off and
on. [Browser, checkout and export receipt](evaluation-panels/video-retimed-compression-app-2026-11-16.json)

Use the portable reader to recalculate interval overlap, the result
totals and both range clocks. With the private rendered media, use the
optional FFprobe and FFmpeg readback to verify the endpoint image hashes.
Restore the pinned sources by adding a local `path` to every item in the
receipt's `sources` list. The builder checks each acquired source hash.

```sh
export PYTHONPATH=backend
receipt=docs/evaluation-panels/video-retimed-compression-2026-11-16.json.gz
python3 -B -S -m scripts.video_speed_compression_panel verify "$receipt"
backend/.venv/bin/python -B -m scripts.video_speed_compression_panel build "$run/held" "$run/sources.json"
python3 -B -S -m scripts.video_speed_compression_panel verify "$receipt" --media "$run/held"
npm --prefix frontend run build
backend/.venv/bin/python -B -m scripts.video_speed_compression_app "$run/held" "$receipt" "$local/paid" --browser
```

Use a filesystem with hard-link support for the paid app replay.
