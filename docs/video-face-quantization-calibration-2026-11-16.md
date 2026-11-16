# Face-swap calibration using coded quantization

Strong recompression reduced the app's ability to flag swaps that
occupy an entire clip. An earlier exploratory result suggested that
combining both readings on each sampled face could help if calibrated
for delivery quality. This prospective study compares that joint
reading with and without a compression measurement. [Earlier face calibration](video-paired-face-calibration-2026-11-16.md)
· [Controlled study of H.264 compression](https://arxiv.org/html/2608.28685v2)

## Measured quality

On the verified picture clock, read the initial luma quantization
parameter from each H.264 picture's slice and parameter set. For an
eight-bit slice, its initial value is 26 plus the picture parameter set's
offset plus the slice offset. The clip input is the median over inter
pictures. This measurement describes the delivered encoding. Earlier
transcodes and individual block values are outside its scope.
The frozen measurements cover one slice per picture. [H.264 (08/2024), equation 7-32](https://www.itu.int/rec/dologin_pub.asp?id=T-REC-H.264-202408-I%21%21PDF-E&lang=e&type=items)
· [FFmpeg bitstream tracing](https://ffmpeg.org/ffmpeg-bitstream-filters.html#trace_005fheaders)

The paired face check uses the same six native positions and pinned
weights for every method. Calculate both normalized readings on each
face, average them, and take the median of at least four jointly
measured faces. Using the 118 measured ordinary deliveries from 59
development groups, compare three calibrations of that same score.

1. Fit a shared scale and operating point across all ordinary deliveries.
2. Keep the shared scale and interpolate the operating point using the
   measured median quantization.
3. Interpolate both checks' centers and scales, as well as the operating
   point, using the measured quantization.

The observed development medians of 21 and 34 are the interpolation
anchors. The measured development range is 13–46. Values within that
range use the nearest endpoint outside the two anchors. The conditioned
arms abstain when the quality reading is missing or outside the range.
Leave each source group out of normalization in turn and take the
90th or 80th percentile of the ordinary scores to set both operating
points. The reference scores and the holdout manifest were locked
before new held model inference. These operating points are for the
research comparison.

## Fresh deliveries and negative transfer

Thirty held targets each supply one original and one publisher-rendered
swap from each of two generators. The fixed order excludes the previous
60 targets, their known donor keys and their source digests. The source
clips may be up to 12 seconds long. The delivery recipe uses four
seconds of matched native pictures, at 640 × 360 and 25 fps. Each clip
has a direct CRF 23 delivery, plus second encodes at CRF 30 and CRF 36.
The intermediate treatment is absent from development. Held scoring uses
the measured quantization directly. Encoder labels are used to report
the results by delivery. [Publisher source and terms](https://huggingface.co/datasets/godmodes/rtfs-10k/blob/bca865e675852666a426368b2f5614abf1c1926a/README.md)

A separate ordinary panel uses all eight previously qualified camera
recordings from VCD2 under those three delivery treatments. It supplies
a cross-dataset false-alarm check. These recordings had prior project
exposure in an earlier study and supply zero calibration observations here.
[Camera source qualification](video-vcd2-matched-controls-v290-result.md)

## Results

The table counts positive flags across 60 swaps and 30 matched originals
per delivery. Each calibration uses its frozen wider operating point.
The fallback rows retain existing positive flags and add candidate flags
on any remaining clips. Three swaps and one original per delivery were
unassessed by all methods. They remain in the denominators.

| Review | Direct swaps | Intermediate swaps | Strongly recompressed swaps | Direct originals | Intermediate originals | Strongly recompressed originals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing paired review | 45 | 30 | 18 | 3 | 6 | 5 |
| Shared calibration alone | 44 | 30 | 22 | 5 | 7 | 4 |
| Existing review plus shared fallback | **48** | **34** | **24** | 5 | 8 | 5 |
| Existing review plus quantization-adjusted cutoff | 48 | 34 | 23 | 6 | 8 | 5 |
| Existing review plus quantization-adjusted scales and cutoff | 48 | 33 | 22 | 7 | 9 | 5 |

The shared fallback supplies 13 additional flags across eight swap
target groups and four additional ordinary alerts across three source
groups. The six additional strongly recompressed swaps include both
generators and span five targets. On those encodes the ordinary alert
count stays at five. The shared fallback adds one strongly recompressed
swap and two ordinary alerts across the three deliveries at its tighter
point. Measured quantization helped describe the deliveries. The two
quantization-adjusted calibrations supplied fewer swap flags and more
ordinary alerts than the selected shared fallback at the wider point.

On the eight camera controls, existing ordinary flags were 4/8 direct,
3/8 intermediate and 3/8 strongly recompressed. The selected fallback
adds zero alerts across those 24 deliveries. Those existing flags show
how difficult it is to transfer the paired review to a new source domain.
All held results concern these renderers, camera sources and delivery
profiles. Other face sizes, codecs and edit lengths require their own
assessment.

A separate FFprobe and FFmpeg comparison verified every picture clock
and quantization value on 270 held deliveries and 24 camera deliveries.
All 60 target-to-swap source pairs had comparable delivery cadence.
The released packet contains sample clocks, scalar readings, parameters
and source digests. Media, face boxes and weights stay local.
[Numeric comparison and source receipt](evaluation-panels/video-face-quantization-calibration-2026-11-16.json.gz)

## Local app

Set `VERITAS_VIDEO_WHOLE_FACE_SAME_FRAME_REVIEW=true` together with
`VERITAS_VIDEO_WHOLE_FACE_REVIEW=true` in development. The fallback
defaults off. It uses the two existing readings and applies to upright
640 × 360 H.264 MP4 with 100 verified pictures on a regular 25 fps clock.
At least four of the six scheduled faces must have both responses.
The app keeps existing flags and shows this new finding when the shared
fallback adds one. It carries zero weight in the overall AI assessment.

Six simulated USD 9.99 checkouts exercised the normal pipeline with the
paired detector isolated. Two fresh strongly recompressed swaps flagged
with the fallback enabled and both had unchanged overall assessments.
The matched ordinary clip stayed unflagged and an existing paired flag
was preserved. No analysis job queued before simulated payment. Both
new findings cover sampled times **0.560–3.400 s** and display the
original frames and face crops at frames **14, 57 and 85**. The API,
PDF and JSON agree, and the signed package verifies those assets against
the original. This coverage range describes the sampled positions.
[Local app verification receipt](evaluation-panels/video-face-same-frame-app-2026-11-16.json)

## Replay

The portable checks recreate the calibration and recount both held
panels. Their output agreed on Python 3.12 and 3.14.

```sh
export PYTHONPATH=backend
receipt=docs/evaluation-panels/video-face-quantization-calibration-2026-11-16.json.gz
python3 -B -S -m scripts.video_face_quality_calibration verify "$receipt"
python3 -B -S -m scripts.audit_video_face_quality replay "$receipt"
```

Follow the [earlier native replay](video-whole-face-temporal-transfer-2026-11-16.md#replay)
to obtain the pinned publisher file index and set `runner`. Recover this
study's 90 items by passing this receipt to the same restore command.
The first two studies supply the earlier deliveries and their paired
face response journals for the fit. All paths below point to private
working directories.

```sh
python3 -B -S -m scripts.audit_video_face_temporal restore "$receipt" "$run/publisher.json" "$run/roster.json"
"$runner" -B -m scripts.video_face_temporal_panel acquire "$run/roster.json" "$run/source"
"$runner" -B -m scripts.video_face_temporal_delivery "$run/roster.json" "$run/source" "$run/pair"
"$runner" -B -m scripts.video_face_quality_inputs middle "$run/pair" "$run/held"
"$runner" -B -m scripts.video_face_quality_inputs measure "$first/delivered" "$run/first-quality.json"
"$runner" -B -m scripts.video_face_quality_inputs measure "$second/delivered" "$run/second-quality.json"
"$runner" -B -m scripts.video_face_quality_inputs measure "$run/held" "$run/held-quality.json"
python3 -B -S -m scripts.video_face_quality_calibration freeze \
  "$first/delivered/manifest.json" "$run/first-quality.json" "$first/reference-dev.jsonl" "$first/reference.jsonl" \
  "$second/delivered/manifest.json" "$run/second-quality.json" "$second/reference.jsonl" \
  "$run/held/manifest.json" "$run/freeze.json"
"$runner" -B -m scripts.video_face_temporal_reference "$run/held" "$run/reference.jsonl" held --device mps
python3 -B -S -m scripts.video_face_quality_calibration report \
  "$run/held/manifest.json" "$run/held-quality.json" "$run/reference.jsonl" "$run/freeze.json" "$run/result.json.gz"
python3 -B -S -m scripts.audit_video_face_quality audit "$run/held" "$run/held-quality.json" "$run/audit.json"
"$runner" -B -m scripts.video_face_same_frame_app "$run/held" "$receipt" "$run/paid-app"
```
