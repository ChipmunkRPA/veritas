# Short repeated mouth movement in one upload

The test searched for two short intervals that show similar moving mouth detail while the surrounding face and concurrent speech differ. The aim was to localize a copied mouth sequence within one upload. It does not estimate which occurrence came first in the editing process or whether either was AI-generated. It also does not cover redraws that create a new mouth sequence.

Prior single-upload work used speech energy, spectral transitions, and visible mouth opening to estimate a signed lag. It missed all 22 known local shifts on CREMA-D, and a separate paired feature method found two of 64 GRID shifts with 17 alerts on ordinary controls. The new test uses both observed mouth intervals as its own comparison. Recent research treats timing and speech articulation as separate questions. The test reuses the existing small face detection asset and applies no new pretrained model. [Earlier CREMA-D timing comparison](video-lipsync-local-delay-transfer-2026-11-11.md) · [Earlier GRID timing comparison](video-lipsync-local-contrast-transfer-2026-11-12.md) · [PVSync](https://arxiv.org/abs/2610.09223) · [AV-SyncBench](https://arxiv.org/abs/2607.00726)

## Frozen comparison

The first study used two GRID speakers and four CREMA-D actors for development. Six different GRID groups and eight different CREMA-D actors supplied the held deliveries. The recordings had earlier project exposure. The test created new two-second H.264 views at 25 fps and verified their native frame timestamps and contiguous 16 kHz audio. Each source supplied an earlier and a later copy of a moving mouth patch, plus a second-generation H.264 version of the later copy. The copied parts are eight or ten frames long. The first-generation views use CRF 22 and the tandem views use CRF 34. The six controls are untouched pictures and sound, tandem compression, a whole-picture loop, a still mouth hold, a gain-only change, and a local copy where the mouth movement and soundtrack repeat together. [GRID source terms](https://zenodo.org/records/3625687) · [CREMA-D source terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt) · [Face geometry asset terms](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

The comparison requires two disjoint 7–12-frame ranges. It looks for a detailed, moving region near the mouth with similar luminance patterns and a similar sequence of changes in both ranges. Nearby face tiles must differ, and each range must contain active sound that differs from its counterpart. A useful result needs at least 60% intersection over union for each interval and the exact unsigned displacement in frames. Development chose 0.28 from four fixed points, with two correct locations and one control flag. The held gate required eight correct locations across four groups per bank, at most one false alert per bank, at least 75% coverage, and zero alerts on whole-picture loops or matched sound and picture copies.

## Held results

| First comparison | GRID | CREMA-D |
| --- | ---: | ---: |
| Assessed views | 54/54 | 72/72 |
| Correct mouth-copy pairs | 0/18 | 3/24 |
| Correct pairs after tandem encoding | 0/6 | 0/8 |
| Ordinary controls with an alert | 1/36 | 2/48 |
| Whole-picture loops with an alert | 1/6 | 2/8 |

The three correct pairs came from separate actors. One report located **0.480–0.760 s** and **1.200–1.480 s**, with a displacement of **720 ms**. Both returned intervals lie within the authored ranges. A separate FFmpeg and FFprobe readback verified all three localized deliveries, including 150 native timestamps, 96,000 source samples and 24 exact frame/crop pairs. The current one-second timing comparator had no eligible full lag window in these two-second fixtures. The current mouth-hold check reported 13 still controls and no moving copy. The three whole-picture alerts showed that differences at nearby face tiles alone could admit a wider copy. The first candidate missed the frozen gate.

## Wider-picture transfer

The follow-up adds a requirement for texture differences well above and below the mouth. It compares the inner pictures of both matched ranges, excluding two pictures at either end. Two separate Wikitongues speakers set the new review point at 0.38. Seven different speakers supplied authored edits and controls. Two additional low-cadence recordings supplied only original and tandem controls. These are spontaneous speakers from previously licensed local source files. Their new two-second deliveries use 25 fps H.264 and exact 32,000-sample PCM. The first preparation found that a seeked Vorbis excerpt returned two extra output samples. The scoring run restarted with an explicit trim at sample 32,000 before setting the point or opening held results. [Sources and licensing](video-mouth-hd-hold-2026-11-12.md)

| Held Wikitongues comparison | Prior nearby context at 0.28 | Wider context at 0.38 |
| --- | ---: | ---: |
| Assessed views | 67/67 | 67/67 |
| Correct mouth-copy pairs | 2/21 | 3/21 |
| Speakers with a correct pair | 1/7 | 2/7 |
| Correct pairs after tandem encoding | 0/7 | 0/7 |
| Ordinary controls with an alert | 1/46 | 2/46 |
| Whole-picture loops with an alert | 1/7 | 2/7 |

A retrospective replay of all 67 deliveries used 0.38 for both methods. Both selected the same three correct pairs and the same two whole-picture loops, with identical ranges and crop coordinates. The wider-picture guard added no observed benefit on this panel. The comparison fell short of ten correct pairs across four speakers and the requirement for zero alerts on whole-picture loops. Both methods remain experimental. The receipts record the two source intervals, native timestamps, and crop hashes for every reported pair. Raw media stays local. [Case results and frame commitments](evaluation-panels/video-mouth-sequence-reuse-transfer-2026-11-12.json)

## Local replay

Run the five phases `freeze`, `development`, `point`, `held`, and `report` with separate new local run directories. The first module takes its run directory. The wider-context module takes its run directory, the archived Wikitongues HD study folder and the first run directory. Use `PYTHONPATH=backend` and the existing backend environment for native media replay. The public case arithmetic also runs without third-party libraries.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_video_mouth_sequence_transfer docs/evaluation-panels/video-mouth-sequence-reuse-transfer-2026-11-12.json
```
