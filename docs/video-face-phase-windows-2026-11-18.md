# Cross-scale phase for locating brief face swaps

A local study tested whether spatial phase adds useful locations to
standalone video analysis. At its frozen operating point, the new check
added **zero correct ranges** in 32 edited deliveries and **one ordinary
alert** in 78 controls. The alert came from an ordinary local face blur.
The app continues to use its existing checks.

## Method and source limits

Recent phase-congruency research combines learned phase filters with an RGB
stream. We used fixed arithmetic on retained face-aligned frames and
compared how phase changed within a plausible edit and in nearby regions.
[Learning phase congruency representations](https://doi.org/10.1016/j.mlwa.2026.100941).
Separate forensic research treats time-range localization as its own
evaluation task.
[Gu et al., 2026](https://openaccess.thecvf.com/content/CVPR2026F/html/Gu_Beyond_Static_Artifacts_A_Forensic_Benchmark_for_Video_Deepfake_Reasoning_CVPRF_2026_paper.html).

The original brief face-swap study provided the retained clips, brief-scan
results and alignment. A fixed
hash ordering selected four groups from that study's fit archive and
eight groups from its confirmation archive. The phase study committed its
source groups, files and method before measuring any phase scores. Group
IDs, exact source and rendered-variant digests and final delivery digests
were disjoint between splits. Both cohorts had earlier project exposure.
Individual identity overlap and third-party pretraining exposure are
unknown. [Original brief face-swap study](video-local-face-intervals-2026-11-16.md)

The source is RTFS-10K at revision
`bca865e675852666a426368b2f5614abf1c1926a`. The dataset card declares
CC BY-SA 4.0 and describes the source footage as Creative Commons.
Underlying participant and portrait attribution remains incomplete.
The local evaluation used retained publisher swap tracks and matched controls.
[Publisher card and terms](https://huggingface.co/datasets/godmodes/rtfs-10k/blob/bca865e675852666a426368b2f5614abf1c1926a/README.md) ·
[Original rights and source review](video-common-source-preparation-v89.md).

Two publisher-generated swap tracks per group had fixed, short appearance
changes inserted at known frame intervals in corresponding ordinary
footage. Controls include an unchanged clip, exposure adjustment, a
jump within the same recording, a matched cut to a different recording
and local face blur. One held group lacks both encodings of its blur
control. Test conditions were upright 640 × 360 H.264 4:2:0 files with
exactly 100 pictures at 25 fps. Direct and strongly recompressed
deliveries are paired observations.

The comparison uses three wavelengths in five aligned face regions,
measured every other picture. The selected arm discounts scores when
three nearby collar regions also change. A range requires face
geometry and continuity on both sides. A useful range overlaps at least
half the union of the predicted and labeled intervals. Archived
landmarks supported alignment without running another detector. The native
held audit checked 11,000 picture timestamps and 4,921 raw and aligned
crop hash pairs.

## Frozen results

Two variants and 11 cutoffs were compared on 16 edits and 40 controls.
The selected collar variant at 0.075 added zero correct development
ranges and zero ordinary alerts. This point was frozen before scoring
the 32 held edit deliveries and 78 held ordinary deliveries.

| Held delivery | Edited assessed | Ordinary assessed | Brief-scan correct | Additional phase correct | Additional phase ordinary |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct | 14 of 16 | 36 of 39 | 2 | 0 | 1 |
| Strong recompression | 14 of 16 | 36 of 39 | 2 | 0 | 0 |
| Both | 28 of 32 | 72 of 78 | 4 | 0 | 1 |

There were zero wrong edit ranges and zero new ordinary cut alerts
at the frozen point. The candidate needed six additional correctly
located edits across at least three groups while adding at most two
ordinary alerts in one group, at most two wrong edit ranges and zero
ordinary cut alerts. It fell short of both recall requirements.

The brief scan is a partial baseline. The earlier study ran its full
worker on a selected set of likely additions. The exact input intersection
contains 26 edited deliveries and two ordinary deliveries. That full worker
already found all 26 edited ranges and flagged both ordinary deliveries.
The phase check added no ranges in that intersection. Its one ordinary
alert is outside the intersection, so the earlier worker's result on
that file is unknown. The intersection cannot estimate full app recall
or false alarm rates on the current held panel.

## Exploratory comparison at matched ordinary-alert budgets

Every distinct measured cutoff was recounted after opening the held
results. These points maximize new correctly located edits within each
ordinary-alert budget. These are exploratory comparisons after the frozen run.

| Maximum extra ordinary alerts | Best new correct ranges | Extra ordinary alerts used | Ordinary groups affected | Wrong edit ranges | Extra ordinary cuts |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 | 0 |
| 2 | 1 | 2 | 2 | 0 | 0 |
| 4 | 1 | 2 | 2 | 0 | 0 |
| 8 | 2 | 7 | 3 | 1 | 0 |
| 20 | 5 | 20 | 7 | 4 | 2 |

The archived brief scan had zero ordinary alerts. Matching that
ordinary-alert count found no improvement. Even at two extra ordinary
files, the one additional location came with ordinary alerts across
two groups, above the original one-group limit. The findings apply to
these short H.264 clips and the fixed computation tested here.

## Independent arithmetic replay

The [numeric receipt](evaluation-panels/video-face-phase-windows-2026-11-18.json)
contains model-free scores and frame-relative ranges, with group IDs
randomized and time origins removed per file. It omits raw media, crop
images, direct file digests, identifiers and source paths. A standard
Python reader reproduces the point selection, decision counts and
exploratory tradeoff without application dependencies or network access.
The numeric replay checks arithmetic. The exact native comparison
requires the retained source files and audited private receipts.

```sh
python3 -B -S backend/scripts/review_video_face_phase_windows.py \
  docs/evaluation-panels/video-face-phase-windows-2026-11-18.json
```
