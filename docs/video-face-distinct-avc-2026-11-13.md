# A second distinct face-change range in four-second H.264 video

Chipmunk Verify can run an optional second search when the Full or Deep
analysis already has a qualified short face range and the existing review
has supplied no second range. Each reported range has its own source
timestamps and before, during and after images. The second range shows
the exact frames used at both boundaries. The browser, Standard PDF and
JSON distinguish a recorded second range, a completed search with no
qualifying second range, and an unassessed search. The existing ranges
and the overall AI assessment are preserved. The measurements describe
appearance changes and contribute zero weight to that assessment.

The trial covers only upright, 120-picture, 1280 × 720, 30 fps MP4 with
verified native presentation timestamps, H.264 `avc1` and 8-bit 4:2:0.
Set `VERITAS_VIDEO_FACE_SECOND_AVC_REVIEW=true` with
`VERITAS_VIDEO_FACE_SHORT_AVC_REVIEW=true` and appearance mode on a local
instance. The older second-range route takes precedence where it works.
The new setting defaults off. The retained comparison supports a
local trial. The source-separated confirmation must establish gains on
different recordings before any broader rollout.
The additional route uses the 120 native motion pictures, with at most
84 scheduled face inputs and 16 edge candidates. When the first range
uses the same bounded route, the two searches share decoded measurements.
When the first range comes from the guarded half-rate check, the
additional bounded pass has a maximum of 12 seconds and respects the
remaining analysis time.

## Method and local results

Both variants use the existing face measurements, edge shortlist and
fixed response point. A second range must be at least three native
pictures away from the first, and cannot duplicate a coarse or refined
range at half temporal intersection over union. The short variant
searches five to seven pictures. The wider variant searches five to
nine. A useful addition reaches at least half temporal intersection
over union with a previously missed labeled interval. The predeclared
comparison requires three useful additions across two source groups and
permits at most one misplaced range and one additional ordinary finding,
with zero added genuine speaker-change or cut findings. It uses the
existing commercially admitted face assets. Consecutive-frame comparison
also appears in recent research on short altered intervals in genuine
footage. [Earlier app review](video-brief-face-avc-clock-confirmation-2026-11-13.md)
· [Face assets and terms](video-appearance-plan-v107-method.md)
· [Short-interval boundary research](https://arxiv.org/abs/2606.07090)

| Retained test, 125 deliveries | Current review | Plus five-to-seven-picture search | Plus five-to-nine-picture search |
| --- | ---: | ---: | ---: |
| Labeled intervals located | 46/62 | 52/62 | 56/62 |
| Two-swap deliveries with both intervals located | 5/16 | 11/16 | 15/16 |
| Ordinary clips with any face-appearance range | 15/79 | 15/79 | 15/79 |

All ten additions from the wider search matched both labeled boundaries
exactly and came from six source groups. Four of the additional
locations were the same eight-picture edit in two encodes of two
different recordings. The remaining six additions were six pictures
long. Both variants produced zero misplaced new ranges. The ordinary
count includes prior coarse face observations and measures review load
on genuine clips, with no AI-origin classification implied. These
measurements reuse previously evaluated sources and controls.

The source-separated test uses 56 newly prepared deliveries from eight
retained publisher captures at 30 or 29.97 fps. It excludes the source
links, known donor keys and capture components from the prior two-range
comparison banks. These recordings have broader project exposure and
the publisher's donor identity metadata is incomplete. Both swap
families appear in both positions. Genuine controls include original
footage, same-recording jumps and cuts to another person. Single-edit
clips pair a swap with a jump or local exposure change.
[Source selection and permissions](video-common-source-preparation-v89.md)

| Source-separated test | Current review | Plus five-to-seven-picture search | Plus five-to-nine-picture search |
| --- | ---: | ---: | ---: |
| Labeled intervals located | 41/48 | 41/48 | 42/48 |
| Two-swap deliveries with both intervals located | 12/16 | 12/16 | 13/16 |
| Ordinary clips with any face-appearance range | 0/24 | 0/24 | 0/24 |

The new route was eligible on 17 of 56 deliveries. Its one exact gain
was an eight-frame range at **2.367–2.633 s** in a tandem-compressed
29.97 fps source. Two versions of one clip with two known edits
supplied no first range, leaving four labeled intervals outside the
second search. Both variants fell short of the predeclared confirmation
gate. The wider variant remains a local opt-in based on the measured
complementary gains. Testing the first-range coverage on new edits
is the next open detection problem.

## Local customer flow

The isolated Full worker used two retained clips with known dual swaps,
one co-occurring genuine jump and one genuine two-cut clip. With the
option enabled, an eight-picture edit appeared at **3.000–3.267 s**
with original frames **89, 94 and 98**. On the other positive, a
six-picture edit appeared at **3.133–3.333 s** with frames **93, 97
and 100**. The earlier range and the overall assessment were preserved.
The API, JSON, Standard PDF and signed original replay verified both
sets of original images and face crops. The jump and genuine cuts
gained no additional range. The same eight on/off runs used simulated
prepaid checkout and queued zero unpaid jobs.
[Local customer-flow receipt](evaluation-panels/video-face-distinct-avc-app-2026-11-13.json)

## Independent replay

Use the Python standard library to rerun the published arithmetic.
The receipts contain source hashes, clocks, positions, scalar responses
and exact image commitments. Source video and face vectors remain local.

```sh
export PYTHONPATH=backend
python3 -B -S -m scripts.video_face_distinct_avc review \
  docs/evaluation-panels/video-face-distinct-avc-retained-2026-11-13.json
python3 -B -S -m scripts.video_face_distinct_avc review \
  docs/evaluation-panels/video-face-distinct-avc-source-separated-2026-11-13.json
```

To verify the original timestamps and images, use FFmpeg and the
local source banks. The separate reader generates an independent
`framehash` stream and checks every clock tick and every selected
original image against the receipt. The local readback checked 21,720
presentation timestamps and 48 selected original frames across 181 clips.

```sh
python3 -B -S -m scripts.review_video_face_distinct_avc_pixels "$receipt" "$run"
```

Native scoring additionally uses the pinned face assets and verified
archived sources from the [carrier study](video-brief-face-120-carrier-2026-11-13.md).
It compares both choices against the earlier app with its new AVC
second-range option off. For a new source-separated run, use a fresh
private directory for `$run`.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VERITAS_VIDEO_FACE_SECOND_AVC_REVIEW=false
backend/.venv/bin/python -B -m scripts.video_face_distinct_avc freeze "$run" --panel fresh
backend/.venv/bin/python -B -m scripts.video_face_distinct_avc prepare "$run"
backend/.venv/bin/python -B -m scripts.video_face_distinct_avc run "$run"
```

Run the local customer-flow check with the retained four-second bank
and a new private output directory.

```sh
backend/.venv/bin/python -B -m scripts.video_face_distinct_avc_app "$retained_bank" "$app_run"
```
