# Measured end context improved coverage and extended control flags

The same rule ran on all six saved clips. Every run verified its
native picture clock and reconstructed the exact observed face crops.
One pass made 240 crop forwards. A second read-only pass reproduced
the complete JSON byte for byte. [Method](video-faceswapai-edge-review-v194-method.md) · [Aggregate](evaluation-panels/video-faceswapai-edge-review-complete-v194.json)

| Cases | Existing last supported image | Proposed last supported image | First measured return |
| --- | ---: | ---: | ---: |
| All four swap renderings | 5.583 s | 5.875 s | 6.083 s |
| Both controls | 7.375 s | 7.667 s | 7.833 s |

The proposed extent overlaps **69 of 72 edited native positions** in
each swap rendering, up from 62. Temporal IoU rises from 0.861 to
0.958. Four appended positions were measured as core-like and three
intervening positions were unmeasured. The return bracket spans four
unmeasured native positions around the known 6-second end. The earlier
“after” image at 5.833 seconds is now identified as core-like. The
6.083-second return supplies the first measured post-edit context.

Both controls also extended their later flags by seven positions,
outside the ordinary recompression interval. Their clips share the
same synthetic source. This test supports showing the measured return
as review evidence. It supplies no basis to interpret the cue as AI
origin or adopt it into overall confidence. Independent camera
controls and additional face-swap sources remain necessary.
