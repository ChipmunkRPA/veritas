# VCD2 camera-original controls

Use Microsoft VCD2 camera footage to test how often a video face-change check
flags camera originals and ordinary edits. Each recording and its edited copy
form one group. This source qualification does not run the check or establish
its sensitivity to face swaps.

Microsoft releases the recordings under CDLA-Permissive 2.0, which permits
use and modification and requires the agreement text to accompany shared
data. The TH category preserves camera pictures without background
post-processing. Microsoft regularized timestamps in the benchmark files
without dropping, duplicating, or interpolating pictures. The raw release
also preserves irregular webcam timing. [Microsoft release](https://github.com/microsoft/VCD/tree/3c87cbdbe68505da6d590cc4ed8c2af73bcc5c68) ·
[Data agreement](https://github.com/microsoft/VCD/blob/3c87cbdbe68505da6d590cc4ed8c2af73bcc5c68/LICENSE-DATA) ·
[Camera-native dataset paper](https://arxiv.org/html/2603.26763v3)

## Selection and commitments

Use the official full and benchmark CSVs, corresponding published file lists,
and agreement text. The script checks all five SHA-256 pins. All 120 benchmark
names join uniquely to the 838 raw recording names by 32-character filename
stem. Retain only the 40 `TH` candidates. Require 720p or 1080p and a valid
camera type. Sort the four spatial and temporal information strata by name.
Within each, rank candidates by SHA-256 of the fixed seed
`chipmunk-vcd2-genuine-20261110`, a colon, and the recording stem. Take the
first three with distinct camera type across the roster. The resulting 12
groups include six at each resolution and four with lighting issues reported
by at least one quality rater. Lighting ratings describe the recording and
do not label its authenticity. [Official metadata](https://vcdpublic.blob.core.windows.net/vcd2/metadata/README.md)

The raw and benchmark filenames identify a recording, and USB vendor and
product codes identify camera types. Neither field identifies people. The
publisher reports 799 participants across 838 recordings, so distinct
recording stems do not establish distinct people. No participant names,
identity embeddings, or cross-dataset identity matching are needed. Keep
the roster, media, exact file receipts and frame clocks on the private
Seagate archive. Report aggregate counts by recording component.

Qualification accepts one roster slot at a time. It requires the pinned
metadata, matching freeze, mounted archive, and available disk space.
It binds the official Azure HEAD and GET using the object ETag and length,
checks the publisher's HTTP Content-MD5 when present, and saves a local
SHA-256 and observed headers. Azure does not publish a source SHA-256 in
the CSVs. The script checks the Matroska and FFV1 headers and geometry and
requires the decoded picture PTS list to match the independent FFprobe
packet PTS list. It records the observed timestamps and the exact planned
intervention bounds. The benchmark summary lists 10-second clips, while
the selected CSV entries report about 15 seconds. Use measured packet and
picture times for all derived artifacts. Each attempt has its own directory.

## Paired control design

For each qualified recording, plan two copies with the same picture count,
geometry and clock. The first is a clean rewrap/re-encode. In the ordinary
edit copy, compress only the pictures whose presentation timestamps fall
in the half-open interval `[3,6)` seconds relative to the first decoded
picture, using H.264 CRF 24. Reinsert those decoded pictures in their original
positions. Both copies use the same lossless final video encode, matching
track handling, metadata policy and source-clock mapping. Record explicit
color conversion for both copies and verify that final decoding produces
the intended picture bytes. Retain PTS, duration, time base, and exact
frame-index bounds from the actual source.

Measure per-frame RGB mean absolute difference on the decoded pairs
independently of any model. Require exact equality before and after the
intervention, and changed picture bytes for at least one selected frame.
Count original flags and ordinary-edit flags per source recording and
report the paired difference with a source-recording interval if enough
groups qualify. This sample measures false alarms in a small webcam cohort.
Microsoft's separately published background-blur and background-replacement
groups can provide additional context if measured later. Those groups use
different recording stems and should not be described as paired TH edits.

## Commands

Run from the repository root. With the pinned official metadata cache,
preflight performs no writes and opens no videos. The last two commands
are for a later authorized qualification run.

```sh
metadata=/private/tmp/chipmunk-vcd2.014eoQ
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_genuine_v195 preflight --metadata-dir "$metadata"
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_genuine_v195 freeze --metadata-dir "$metadata"
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_genuine_v195 qualify --metadata-dir "$metadata" --slot 1
```

If needed, recreate the cache by downloading only the five URLs pinned
in the script into a new temporary directory. Use `--attempt 2` or a
higher unused attempt number for a failed slot. Retain any partial bytes
and prior receipts in their original directories. Only `qualify` requests
one video from Microsoft. No command downloads a model or runs inference.
