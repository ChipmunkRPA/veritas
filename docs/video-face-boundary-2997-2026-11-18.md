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

## Other exact MP4 clocks

On a 29.97 fps video, the same presentation time can be stored using
several MP4 track time bases. In addition to the original 30 kHz clock,
the check can use verified 60, 90, 120 and 180 kHz clocks. The report
keeps the time base and integer frame timestamp from the submitted
video. For example, frame 37 is tick 37,037 at 30 kHz and tick 111,111
at 90 kHz. Both give 1.234567 seconds when rounded for display.

A separate container reader checks the declared video duration, the
decode and presentation timing tables, and any MP4 edit list. The video
needs the same zero-origin sequence from the original file. The reader
withholds the check when an edit list changes the playback rate or
extends the display interval, even if the decoded frame timestamps
appear unchanged. Saved packages with the earlier evidence record
continue to verify using its recorded rules. Four previously signed
29.97 fps packages verified against their originals.
[Sample durations](https://developer.apple.com/documentation/quicktime-file-format/time-to-sample_atom/time-to-sample_table)
· [Track edit-list structure](https://developer.apple.com/documentation/quicktime-file-format/edit_list_atom/edit_list_table)

Sixteen retained source files covered staged edits, ordinary controls,
and eight camera-original exports. Five alternate deliveries of each
retained the same H.264 packets and all 9,600 decoded frames, their
presentation times and durations. The alternatives were one remux per
time base and a separate 90 kHz signed-composition variant. The reader
withheld 24 clock controls and two separately tested extended-ending
controls. These results measure file and clock transfer on the same
footage. [Original staged clips](video-face-focus-context-transfer-2026-11-18.md#reproduce-the-counts)
· [Camera originals](video-vcd2-unseen-controls-v287-method.md)

The detector reproduced every record across the 80 alternate deliveries,
including correct locations on all four staged edits per variant, one
ordinary cut left unassessed and no ordinary highlights. The same strict
reader admitted all 176 previously tested canonical-clock clips. Two
simulated paid replays checked a qualified 90 kHz clip and an edit-list
control. Browser, PDF and JSON showed frame 37 and supporting frames 36
and 49 for the edit. The altered edit-list control showed the timing
check as unassessed. Both signed packages verified against their original
files. [Clock-transfer and local application receipt](evaluation-panels/video-face-boundary-2997-carriers-2026-11-18.json)

An independent reviewer can construct a 90 kHz delivery from a qualifying
30 kHz original without encoding the video again. The detector still
reads the uploaded file's native tables. The original and remux can be
checked against their own signed evidence packages.

```sh
ffmpeg -i original.mp4 -map 0 -c copy -video_track_timescale 90000 \
  -movie_timescale 90000 -use_editlist 1 alternate.mp4
```
