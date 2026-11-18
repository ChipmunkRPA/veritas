# Can a single upload reveal a held eye region?

The existing native hold checks compare the whole picture or a region
near the mouth. This test asks whether the same measurements can locate
a temporarily held eye region while nearby face pixels keep changing.
The measurements are limited to the pixels in this specific edit.
[Localized facial editing research](https://arxiv.org/abs/2503.22121) ·
[Facial temporal-coherence research](https://openaccess.thecvf.com/content/ICCV2021/papers/Zheng_Exploring_Temporal_Coherence_for_More_General_Video_Face_Forgery_Detection_ICCV_2021_paper.pdf)

## Source and method

Thirteen publisher captures supplied three development groups and ten
held groups. Selection excluded known source links, published file
identifiers, and file hashes from earlier RTFS work. Exact identity
across different publisher recordings remains unknown. The preparation
chooses the first of three frozen source offsets with measurable
single-face geometry. All 13 groups supplied an eligible excerpt. Each
delivery has 50 upright 480 × 360 H.264 pictures at exactly 25 fps.
[RTFS-10K publisher and CC BY-SA 4.0 terms](https://huggingface.co/datasets/godmodes/rtfs-10k/blob/bca865e675852666a426368b2f5614abf1c1926a/README.md)
· [Pinned face locator and commercial-use license](https://github.com/opencv/opencv_zoo/tree/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet)

The positives are a held image-left eye from 0.640 to 1.080 s, a
held image-right eye from 1.120 to 1.640 s, and a separately compressed
version of the second edit. Each uses earlier pixels from the same
recording and feathers the outer boundary. Ordinary controls retain
their natural eye movement or add recompression, a whole-picture
hold, a picture jump, a local blur, or a local brightness change.
These staged holds give precise source coordinates and durations.
Accuracy on AI eye redraws remains an open question.

The candidate keeps the deployed mouth-region thresholds for
short-term pixel stability, sustained movement in the surrounding
face, and change before and after the candidate range. It adapts the
search geometry to each eye separately and requires the region to
remain near that eye on each of the three exhibit frames. A useful
result requires 0.50 temporal intersection over union and at least
75% of the proposed crop area inside the authored eye patch. The
frozen rule required 15 useful results across six held groups and
at most two new ordinary flags.
[Measured rows and frozen source commitments](evaluation-panels/video-eye-hold-2026-11-18.json.gz)

## Held comparison

| Delivery | Existing native hold checks | Added useful locations | Exact endpoints |
| --- | ---: | ---: | ---: |
| Image-left hold | 0/10 | 3/10 | 2/10 |
| Image-right hold | 0/10 | 0/10 | 0/10 |
| Image-right hold after recompression | 0/10 | 0/10 | 0/10 |
| Total | 0/30 | 3/30 | 2/30 |

The candidate assessed 29 edited clips and 58 of 60 controls. It
flagged one local blur and one local brightness adjustment. The
20 unmodified and recompressed originals and the 20 whole-picture
hold or jump controls acquired no alerts. A separate FFmpeg decode
matched all 15 before, during and after frame hashes for the five
findings and all 15 source crop hashes. The median added review time
was 0.068 seconds after source decode.

The candidate stays in evaluation. It missed most authored holds
even on the measured files. The two direct conditions combine a
different eye with a different placement and duration. The results
cannot isolate which of those differences affected the findings.

## Reproduce

```sh
PYTHONPATH=backend python3 -B -S -m scripts.video_eye_hold_panel verify \
  docs/evaluation-panels/video-eye-hold-2026-11-18.json.gz
```

The package-free command checks range and crop arithmetic, native
clock positions, source commitments, and summary counts. Native
replay also needs the source files, the pinned local face locator,
FFmpeg, NumPy, and OpenCV. Raw media and image crops stay local.
