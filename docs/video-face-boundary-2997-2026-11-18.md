# Face-transition evidence on a 29.97 fps source clock

The local Full face-transition check can now review qualified 29.97 fps
AVC MP4 clips. It uses the same appearance measurement, operating point
and frame sampling as the 30 fps check. The delivery must contain 120
upright 640 × 360 pictures and have a verified zero-origin presentation clock.
Each picture advances 1,001 ticks at a time base of 1/30,000 seconds.
The evidence uses those exact timestamps and includes the last frame
before the transition, the first changed frame and a later supporting
frame, each with an original face crop.
[30 fps comparison and licensed weights](video-face-single-boundary-30fps-2026-11-18.md)

Eight source groups from the held focus study supply 176 staged and
ordinary deliveries, divided equally between direct and more compressed
second encodes. The study changes the delivery clock and preserves the
authored frame order. It uses one-sided prefix and suffix face swaps,
face swaps that coincide with focus changes, and ordinary originals,
focus and gain changes, jumps and cuts. The detector receives one
submitted clip at a time. The group and edit locations are joined after
scoring. [Held comparison, source terms, and downloadable decisions](video-face-focus-context-transfer-2026-11-18.md#reproduce-the-counts)

| Result | Direct encode | Second encode |
| --- | ---: | ---: |
| Standalone swaps located at the exact frame | 16/16 | 16/16 |
| Swap with simultaneous focus change at the exact frame | 8/8 | 8/8 |
| Ordinary clips flagged | 4/64 | 4/64 |
| Ordinary clips unassessed | 14/64 | 8/64 |

Every decision and localized measurement matched the earlier offline
readout when rerun through the application's detector. An independent
container reader verified all 21,120 source picture timestamps. The
ordinary highlights were two whole-picture sharpening and two central
defocus controls per encode. All 16 cross-recording cuts were unassessed.
Six direct ordinary files lacked a precise native boundary.

An additional test used eight camera originals from the separately
qualified VCD2 roster. Each became a direct export and a second encode
at the supported size and cadence. No transition was reported on the 15
eligible deliveries. The check skipped one direct export that kept a
different pixel format. A separate verifier rehashed all eight originals
and all 16 exports, then checked every presentation timestamp and packet
duration. The 16 exports represent eight camera recordings.
[Original camera groups and data terms](video-vcd2-unseen-controls-v287-method.md)
· [Camera-native study](https://arxiv.org/html/2603.26763v3)

The previous serving path skipped this face-transition check at 29.97 fps.
Focus changes and local blur can also produce a face appearance transition.
The highlight carries zero AI confidence weight. These counts cover
short, staged edits and two encodes of the same source groups. Accuracy for
longer videos and other types of face swaps remains unmeasured.

## Local checkout and reports

Five local analyses completed after simulated USD 9.99 payments. The
same edit was submitted with the new option off and on. Its overall
assessment stayed the same, and the enabled check added frame 37 at
1.234567 seconds, with source frames 36, 37 and 49. The remaining cases
were an unchanged clip, an ordinary cut and an ordinary focus shift.
The browser, PDF and JSON showed consistent source positions. All five
signed packages verified against their original videos. Chromium loaded
twelve original frames and crops across the two highlighted cases.
[Local application and original-file receipt](evaluation-panels/video-face-boundary-2997-app-2026-11-18.json)

## Try it locally

Enable the optional development Full check with

```sh
VERITAS_VIDEO_FACE_SINGLE_BOUNDARY_REVIEW=true
VERITAS_VIDEO_FACE_SINGLE_BOUNDARY_NATIVE2997_REVIEW=true
```

Variable frame cadence, shifted time origins, rotated pictures and clips
outside this profile remain outside the check. Saved 25 and 30 fps evidence
uses its original file and clock rules. Seven saved 25 fps and five saved
30 fps signed packages reverified against their original videos.
