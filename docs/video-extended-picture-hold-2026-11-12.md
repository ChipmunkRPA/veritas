# Repeated pictures later in a single video upload

The local video evidence view can now show exact repeated pictures beyond
the old opening limit of 120 frames or 384 MiB of decoded RGB. The check
uses the first video stream. It returns the longest eligible interval and
shows its original image, native frame indices and timestamps. It supplies
zero weight to the AI estimate. Still cards, recording interruptions and
ordinary editing can all produce identical decoded pictures.

## Coverage and replay

The extension requests one consecutive native prefix capped by the first
30 source seconds, 900 frames and 1.5 GiB of total RGB processed, whichever
comes first. The per-frame ceiling remains 32 MiB. The 1.5 GiB ceiling
bounds work across the prefix. Decoding and hashing stream the frames and
keep no frame buffer proportional to the video length. The existing
eight-second decoder watchdog still applies. If the larger scan stops
early, a completed result for the original opening scope is retained.

Each candidate run needs identical display-oriented RGB at every frame,
verified consecutive native timestamps no more than 125 ms apart, and a
first-to-last span of at least 500 ms. A missing or discontinuous source
time splits a run. The report shows the inspected prefix and its uncovered
suffix. An exact repeat that crosses the scan boundary ends at the last
verified frame. A negative result applies only to the inspected scope.

The signed package carries the original frame-table commitment and the
versioned `extended-native-prefix/1` coverage record. With the original
video, the package verifier re-decodes every frame in that recorded scope,
recomputes the clock breaks and longest exact run, and verifies both
endpoint images. Earlier packages retain their original coverage contract.

## Local validation

The roster was fixed before running the comparison. It uses six retained
VCD2 recordings from the separately prepared VCD2 delivery bank, six new
later holds authored on those recordings, and six ordinary hard cuts to
mirrored moving footage. Each webcam also supplies an unchanged 480 × 270
delivery and its native 720p or 1080p control. The 480 × 270 comparisons
share the same lossless RGB H.264 settings. The native controls keep their
earlier H.264 encodes. Synthetic controls exercise a nonzero clock origin,
variable frame intervals, a timestamp break, a clipped hold, a hold after
the budget, VP9 lossless and H.264 CRF 23. [VCD2 delivery preparation](video-vcd2-delivery-v206-method.md) · [Dataset terms](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA)

| Condition | Clips | Exact repeats in old scope | Exact repeats in extended scope |
| --- | ---: | ---: | ---: |
| Authored late holds on webcam recordings | 6 | 0 | 6 |
| Unchanged and ordinary-cut webcam controls | 12 | 0 | 0 |
| Native 720p and 1080p webcam controls | 6 | 0 | 0 |
| Synthetic delivery, clock and budget cases | 6 | 0 | 4 |

Both the variable-clock lossless H.264 case and lossless VP9 case recovered
the exact authored intervals. Lossy H.264 preserved a shorter exact core
at frames 174–191 of the authored 170–194 hold. The 240 ms clock break
prevented joining two individually too-short repeats. The 30-second
boundary clipped the reported 580–619 hold to frames 580–600 and excluded
the later hold beginning at frame 622. Two webcam repeats started one
frame before the authored interval, where the converted source already
had the same picture. These are exact repetition measurements on matched
controls and authored holds.

Independent readback matched all 8,235 native pictures and PTS values in
the 30 source files. The extended pass inspected 8,029 frames in total,
compared with 3,488 in the opening scope. The largest scan processed
258 native 1080p frames, or 1,530.6 MiB of decoded RGB work, in 2.334 s
on the local Mac.

An isolated paid-flow app test used simulated payment and localized frames
347–363 at 17.350–18.150 s. The check inspected 601 frames through 30 seconds
and disclosed the remaining 39 frames. Chrome showed both source endpoints
and sought to their times. The PDF and JSON agreed with the app, and the
signed export replayed against the supplied source file.

## Reproduce the panel

The synthetic panel needs FFmpeg with the RGB H.264 and VP9 encoders and
the project's Python dependencies. The independent comparison uses
FFmpeg's raw RGB SHA-256 framehash output and checks every output PTS against
the original frame table. The runner records each outcome before removing
the temporary authored file.

```sh
cd backend
.venv/bin/python -m scripts.video_extended_picture_hold /tmp/picture-hold-replay
```

To include the six webcam groups, repeat `--webcam FILE` for the unchanged
VCD2 delivery files in slots 01, 02, 04, 07, 08 and 11, in that order.
The runner reads the webcam files in place and records only measurements.
[Aggregate receipt](evaluation-panels/video-extended-picture-hold-2026-11-12.json)
