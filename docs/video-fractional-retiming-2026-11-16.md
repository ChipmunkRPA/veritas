# Locate footage reused at 3:2 and 4:3 playback speeds

Chipmunk Verify can locate footage repeated within one upload at 1.5× or
1⅓× playback speed. It records both time ranges, the relative speed and
the original frames at all four endpoints. The evidence describes reuse
or retiming, which can result from ordinary editing. It carries zero
weight in the overall AI assessment.

## Method and scope

The two established 2:1 comparisons still run first. If both finish
without a match, the additional comparison searches for two disjoint
ranges. It compares every third native frame with every second frame,
and every fourth with every third. It searches every starting phase at
each stride and requires matching direction and magnitude of picture
changes. The method reuses the recompression-tolerant whole-picture
luminance descriptor and its existing settings. It requires at least
13 transitions between checked frame pairs and at least eight with
matching movement. This search uses no additional model weights.
[Earlier 2:1 comparison](video-retimed-compression-2026-11-16.md)
· [Temporal transformations in video copy detection](https://arxiv.org/html/2501.11171v1)

The development setting `VERITAS_VIDEO_FRACTIONAL_RETIMED_SEQUENCE_REVIEW`
runs alongside `VERITAS_VIDEO_RETIMED_SEQUENCE_REVIEW`. The comparison
covers the opening 225 verified frames of an upright 640 × 360 H.264
MP4 at a regular 25 fps in 8-bit 4:2:0. The tested edits skip or duplicate
frames at a fixed speed. Speed ramps, interpolation, other ratios and
other delivery profiles need separate verification.

## Local comparison

Twelve distinct VCD2 camera recordings supplied 300 test deliveries.
Four camera recordings were newly acquired and eight were previously
archived controls. All twelve are distinct from the 12 sources in the
previous recompression retiming panel. Each rate was tested in both
time orders and with the faster or slower segment copied. The test
compared direct H.264 CRF 25 uploads and those same bytes delivered
through a second encode at CRF 33. Source hashes, expected edits,
settings and code digests were saved before measuring the results.
[Pinned publisher terms](https://raw.githubusercontent.com/microsoft/VCD/3c87cbdbe68505da6d590cc4ed8c2af73bcc5c68/LICENSE-DATA)

A correct finding requires at least 0.50 intersection over union on
each labeled range and the correct speed ratio.

| Deliveries | Current check | With 3:2 and 4:3 |
| --- | ---: | ---: |
| 3:2 edits located after one encode | 0/48 | 35/48 |
| 3:2 edits located after two encodes | 0/48 | 29/48 |
| 4:3 edits located after one encode | 0/48 | 41/48 |
| 4:3 edits located after two encodes | 0/48 | 35/48 |
| Ordinary controls flagged | 0/72 | 2/72 |

The new check located 140 of 192 edits, with all four exact endpoints on
94 of them. Four additional edited deliveries returned a location that
missed the overlap or ratio requirement. The two ordinary alerts were
the same apparent repeat in two versions of one archived recording.
The same source also produced one extra alert among 36 earlier repeat
controls. All 29 earlier repeat alerts were preserved. The four newly
acquired recordings yielded 53 correct
findings across 64 edited deliveries and zero alerts across 24
ordinary deliveries. The variants of each recording share source
content, so these per-delivery results are not population error rates.
An independent FFprobe and FFmpeg readback reproduced all 696 reported
retiming endpoint timestamps and RGB image hashes.
[Per-delivery results](evaluation-panels/video-fractional-retiming-2026-11-16.json.gz)

## Application replay

The regular worker, browser and standard reports were tested with six
simulated USD 9.99 purchases. Zero jobs queued before payment. The
replay isolates video reuse checks from the learned detectors. It
compares the same edit with the new
setting on and off, tests a second-encode 4:3 edit, preserves a 2:1
finding and checks two ordinary controls. The API, JSON and PDF share
the original range clocks and frame images. Signed package verification
reproduces the checks against the supplied original. The recompressed
4:3 case shows **1.360–3.040 s** and **5.640–7.880 s**. Selecting its
second range in the browser seeks to **5.640 s**. The replay also checks
that an unsupported speed ratio fails evidence validation.
[Browser, checkout and export receipt](evaluation-panels/video-fractional-retiming-app-2026-11-16.json)

## Replay

To reconstruct the test inputs, add a local `path` to each acquired
recording in the result's `sources` list and save the list as
`sources.json`. The builder checks the SHA-256 of each publisher file.
Keep the raw recordings and test uploads on the test machine. The
portable reviewer recounts matches and verifies the published frame
indexes, source ticks and support offsets. Native verification needs
the reconstructed uploads and a compatible version of FFmpeg.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
receipt=docs/evaluation-panels/video-fractional-retiming-2026-11-16.json.gz
module=scripts.video_fractional_retiming_panel
python3 -B -S -m "$module" verify "$run/clips" "$receipt"
backend/.venv/bin/python -B -m "$module" build "$run/clips" "$run/sources.json"
python3 -B -S -m "$module" verify "$run/clips" "$receipt" --verify-native
npm --prefix frontend run build
backend/.venv/bin/python -B -m scripts.video_speed_compression_app "$run/clips" "$receipt" "$local/paid" --browser
```

Use a filesystem with hard-link support for the app replay output.
