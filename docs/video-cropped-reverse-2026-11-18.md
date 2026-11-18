# Locate footage that appears cropped and in reverse order

The existing checks find cropped footage when it repeats in the same order
and reversed footage when the framing stays the same. This study asks
whether a single upload contains two portions that match after accounting
for both changes. Such a match can be produced by ordinary editing and
carries zero weight in an AI assessment.

## Method and sources

The comparison checks the first 225 frames of 640 × 360 H.264 MP4 at 25
frames per second. It requires a verified, regular presentation clock and
8-bit 4:2:0 pictures. It compares a fixed crop of the wider occurrence
with the full narrower occurrence in reverse order. Both portions must
contain matching moving picture changes for at least half a second and
occupy separate time ranges. The broad comparison uses both the
earlier centered crop presets and off-center proposals from visual
landmarks. This uses ordinary image processing and requires no model
weights. [Spatial video copy localization research](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01818.pdf)
· [Temporal localization evaluation research](https://openaccess.thecvf.com/content/CVPR2022/html/He_A_Large-Scale_Comprehensive_Dataset_and_Copy-Overlap_Aware_Evaluation_Protocol_for_CVPR_2022_paper.html)

The held deliveries were staged from eight separate public camera
recordings. They show dancing, escalators, water, wildlife, a mechanical
demonstration, cycling, and aerial motion. Two additional sources were
used to confirm the runner before the held results. The receipt pins
each exact publisher file, its author and license, the authored range
boundaries, and the measured frames. All media used for the comparison
allows commercial use under its recorded Creative Commons terms. The
published receipts contain timings, coordinates, and hashes. Edited
clips and frame images remain local.
[Broad numeric receipt and complete source credits](evaluation-panels/video-cropped-reverse-2026-11-18.json.gz)

## First held test

The comparison was frozen before scoring. It measures localization as at
least 0.50 intersection over union in both source ranges, the correct
wider occurrence, and a crop with all four edges within two working
pixels of the authored area on a 192 × 108 grid. It preserves existing
crop results and can add a reversed crop when those checks are empty.

| Staged repeat | Existing repeat checks | Added useful range and crop | Exact endpoints |
| --- | ---: | ---: | ---: |
| Centered crop | 0/8 | 6/8 | 5/8 |
| Fixed off-center crop | 0/8 | 4/8 | 2/8 |
| Off-center crop between presets | 0/8 | 2/8 | 2/8 |
| Between-preset crop after a second encode | 0/8 | 0/8 | 0/8 |
| Total | 0/32 | 12/32 | 9/32 |

The broad comparison added zero alerts on 40 unchanged, recompressed,
in-place reframing, in-place reversal with a crop, or held-picture
controls. It also added zero alerts on 16 forward crop or uncropped
reverse controls. The existing checks located 5 of 8 forward crop
controls and 6 of 8 uncropped reverse controls. The twelve added findings
spanned six of eight source recordings. An independent decode reproduced
their 48 original endpoint image hashes. The median time for the added
comparison was 0.260 seconds on this Mac, after decoding.

The broad method stayed experimental. It found 12 of the 32 edited
deliveries, below the frozen 20-case adoption gate. In particular, the
observed results support examining centered crops separately. They give
no evidence of reliable performance after a second lossy encode.

## Independent centered-crop test

A second frozen test used only centered crop proposals and eight
different public recordings. The three staged crops retain 80%, 85%,
or 90% of the reference picture along each axis. Fifteen deliveries
place the wider occurrence first and nine place it second. The same
time and crop accuracy rules apply.
[Centered-crop receipt and source credits](evaluation-panels/video-cropped-reverse-centered-2026-11-18.json.gz)

| Centered crop | Existing repeat checks | Added useful range and crop | Exact endpoints |
| --- | ---: | ---: | ---: |
| 80% | 0/8 | 5/8 | 4/8 |
| 85% | 0/8 | 5/8 | 4/8 |
| 90% | 0/8 | 5/8 | 4/8 |
| Total | 0/24 | 15/24 | 12/24 |

The centered method added zero alerts on 40 ordinary or in-place edit
controls and 16 reverse-only or crop-only controls. The 15 findings
came from five source recordings. The result cleared the frozen minimum
of 14 useful matches, but missed the separate requirement of six
distinct source recordings with a match. This panel alone did not
qualify the local check.
An independent decode matched all 60 source endpoints. The median
added comparison time was 0.049 seconds after decoding. These staged
results show how the test handled a fixed framing change. They do not
estimate detection rates for undisclosed real-world edits.

## Confirmation on another eight camera recordings

The same centered method, controls, and gates ran on a third disjoint
roster. The new captures show a passing train, farm equipment, a
speedboat, two separate football games, a bee, park aerial footage,
and waves. Existing comparisons located 0 of 24 compound edits. The
centered method found both ranges and the right crop in 18 of 24 cases,
with 10 exact locations. It found six of eight edits at each preset
size and six distinct capture groups. None of the 40 ordinary or
in-place edit controls or the 16 simpler repeat controls acquired a
new alert. All 72 endpoint picture hashes agreed with a separate
decode. The median comparison took 0.042 seconds after decoding.
[Confirmation receipt and source credits](evaluation-panels/video-cropped-reverse-confirmation-2026-11-18.json.gz)

This confirmation met the same frozen gate. The optional local
`VERITAS_VIDEO_CROPPED_SEQUENCE_REVIEW` setting now adds the validated
80%, 85%, and 90% centered reverse comparisons after the existing
forward-crop comparison finds no match. It uses the opening 225
pictures of the qualified 640 × 360 MP4 profile. Off-center compound
edits and other formats remain outside this check. The browser
identifies both ranges by their position and which one has wider
framing. The evidence displays four exact source frames, two exact
crops from the wider range, and the matching time ranges. This
observation carries zero weight in the AI estimate.

## Local app verification

Six analyses ran after simulated USD 9.99 checkout. No analysis
queued before payment. One checked clip placed the wider range
first, with **1.400–2.160 s and 5.960–6.720 s**. The other placed it
second, with **0.960–1.960 s and 5.640–6.640 s**. The Standard PDF,
JSON export, and browser displayed the same four frames and two
crops for both cases. The browser sought to the recorded positions
in the original videos. Neither in-place control produced a
compound finding. The signed packages verified against their
original files, and altered crop coordinates or a removed
reverse-order field failed source-evidence validation.
The paired runs produced the same AI estimate.
[Local app replay receipt](evaluation-panels/video-cropped-reverse-app-2026-11-18.json)

## Replay

The package-free command checks the frozen cases, the frame and clock
references, and the correctness of the reported counts.

```sh
PYTHONPATH=backend python3 -m scripts.video_cropped_reverse_panel verify \
  docs/evaluation-panels/video-cropped-reverse-2026-11-18.json.gz
PYTHONPATH=backend python3 -m scripts.video_cropped_reverse_panel verify \
  docs/evaluation-panels/video-cropped-reverse-centered-2026-11-18.json.gz
PYTHONPATH=backend python3 -m scripts.video_cropped_reverse_panel verify \
  docs/evaluation-panels/video-cropped-reverse-confirmation-2026-11-18.json.gz
```

To rerender, recover the publisher downloads from the receipt and store
them in one local folder as `<source-id>.webm`. Check each SHA-256 against
the listed source first. Write `source_roster` from the receipt as a JSON
file, and use the runner's `freeze` and `score` commands in an environment
with FFmpeg, Pillow, NumPy, and OpenCV. Pass `--center-only` to `freeze`
for the centered follow-up.
Exact endpoint counts were recomputed from the saved numeric rows to
correct a tuple-versus-list comparison in the first rollup. Candidate
decisions, useful-location counts, and the gates are the original run.
