# Short face-change review on HEVC and AV1 at 360p30

The prior 360p study used exact 25 fps H.264 conversions. This study tests whether the same optional range review transfers to exact 30 fps through the application's face detection, tracking and appearance paths, with separate HEVC and AV1 deliveries. [Prior H.264 study](video-short-return-encoded-360p-2026-11-11.md)

## Frozen comparison

Three complete native 30 fps face-swap source groups, two at 720p and one at 1080p, each supply two 11-picture swap inserts, an original, and matched illumination and blur edits. Their known original and donor keys are disjoint from the four preceding short-return panels. The first two source groups also supply two ordinary whole-picture cuts. Both edit placements are fixed before scoring and use exact half-open ranges of 2.800–3.167 s and 3.033–3.400 s on the output clock. The source cohorts had other prior project exposure. [Native input preparation](video-native-inserts-v95-method.md) · [Publisher and source rights review](video-common-source-preparation-v89.md) · [Admitted face weights](video-appearance-plan-v107-method.md)

Six naturally recorded CREMA-D originals have distinct actor codes and a 480×360 picture grid. Each delivery pads the sides by 80 pixels, retains every source picture without spatial resampling and assigns an exact 30 fps output clock. Source selection uses only the source-clock deviation and actor code. The maximum permitted movement of a source-relative timestamp is 4 ms. The originals have a millisecond picture clock. The experimental inputs are silent video deliveries. [CREMA-D publisher terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt) · [Native camera-source verification](native-mouth-clips-v120-complete.md)

Each codec gets 23 MP4 files with the same authored pictures and time ranges. HEVC uses libx265 at CRF 25. AV1 uses libaom-av1 at CRF 30. Both use YUV 4:2:0 and verify the actual encoded codec, complete exact 30 fps picture timestamps and decoded RGB commitments. These settings are delivery variants. The test does not equate their bitrates or visual quality.

The app preserves the existing longer face appearance range and tests one separate brief range under the unchanged 0.10 response, sub-half-second duration and 0.50 dedupe IoU rules. Correct localization requires IoU of at least 0.50 against the authored interval. Both codec arms must add at least three correct localizations across at least two source groups, assess at least 80% of swaps and ordinary controls separately, add at most one ordinary control range and add no hard-cut range. Simulated payment exercises the normal paid analysis worker.

## Results

| Outcome | HEVC | AV1 |
| --- | ---: | ---: |
| Correctly localized swaps, incumbent → combined | 6/6 → 6/6 | 6/6 → 6/6 |
| Supplemental ranges correctly located on already-detected swaps | 3 | 5 |
| Matched original/light/blur controls with a false range, incumbent → combined | 3/9 → 3/9 | 3/9 → 3/9 |
| Natural camera controls with a false range | 0/6 | 0/6 |
| Hard-cut controls with a false range, incumbent → combined | 1/2 → 1/2 | 1/2 → 1/2 |
| Assessed by the supplemental check, swaps and ordinary controls | 6/6 and 14/15 | 6/6 and 14/15 |

The supplemental check added zero correct cases and zero false ranges. Its same-shot window guard lacked enough context on one natural camera recording in both codecs. A retrospective comparison of the eight cases with two correct ranges found two AV1 ranges with higher IoU and six with lower IoU, including all three HEVC cases. Both codecs had a median IoU change of −0.017. The exact-30 expansion missed the frozen localization requirement and stays experimental.

The independent verifier matched 6,884 decoded delivery pictures and all 48 selected frame/crop PNGs to the encoded source files. It also reopened all 382 pictures in the six original camera recordings. The maximum movement from original relative camera PTS to the authored 30 fps grid was 2.33 ms. [Aggregate and pixel audit](evaluation-panels/video-short-return-codec-transfer-2026-11-11.json)

## Private report replay

An isolated off/on replay analyzed a single known swap and a recorded camera control in each codec. The app gave the same confidence decision and preserved the incumbent range in each pair. For both deliveries the experimental cue marked **3.133–3.500 s** against the known **3.033–3.400 s** edit, with 0.571 IoU. The three source frame and crop pairs matched the uploads in the evidence API, retained JSON and PDF. Both signed packages passed original-source replay. The camera controls had no range. Checkout and unpaid webhook delivery queued no analysis. Each simulated paid event queued one analysis despite repeat delivery.

## Reproduce

Use the retained archive and a fresh private output directory. The preparation binds the earlier source rosters and matches all source file and frame digests. The independent auditor reopens the encoded files, checks their exact source clock and crop pixels and recomputes the comparison.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_short_return_codec_transfer freeze "$run_dir"
backend/.venv/bin/python -B -m scripts.video_short_return_codec_transfer prepare "$run_dir"
backend/.venv/bin/python -B -m scripts.video_short_return_encoded_app "$run_dir" --codec-transfer
backend/.venv/bin/python -B -m scripts.review_video_short_return_codec_transfer "$run_dir"
```
