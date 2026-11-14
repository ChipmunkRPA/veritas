# Finding a second face change in a side-by-side clip

The local trial can report two separate brief changes in the same side
of a split-screen video. It retains the first result and checks its
already measured face track for another range. Each range has exact
source timestamps, original frames and crops from before, during and
after the change, and motion measurements at both endpoints. Appearance
changes can have several causes and carry zero weight in the AI score.
[Earlier track comparison and commercial weight references](video-face-split-track-2026-11-13.md)

## What the trial measures

The first pass chooses the tracked side and strongest eligible range.
The follow-up reuses that side's native pictures, face measurements,
boundary shortlist and off-face motion check. It checks spans of five
to nine pictures with a minimum three-picture gap from the first span.
The deadline is shared with the first pass. When the follow-up times
out, the first range is retained and the follow-up is marked unassessed.
The app, PDF, JSON and signed original replay preserve both ranges and
their shared source evidence. The second range may occur earlier in
playback if the first range was stronger.

## Paired local result

We reused the 64 delivered clips in the earlier side-by-side study.
Twenty-four contained two staged face swaps in the same view. Forty
matched controls included ordinary resize, guest cuts and camera cuts.
The 64 clips reused eight recorded source groups. These figures describe
this already used panel.

| Measure | First range only | First plus follow-up |
| --- | ---: | ---: |
| Exact localized intervals | 24/48 | 46/48 |
| Edited clips with both boundaries of both swaps located | 0/24 | 22/24 |
| Ordinary control clips with a reported range | 0/40 | 0/40 |

The additional intervals matched both known boundaries and the correct
side. All eight source groups contributed new locations.
[Paired result and picture commitments](evaluation-panels/video-face-second-pane-2026-11-14.json)

We also re-encoded 16 derived clips with just one of the two swaps
retained. The first pass found 15 exact ranges and missed one. The
follow-up reported no extra ranges in the 15 clips where it ran.
The original-byte replay checked 9,600 presentation timestamps and
183 displayed frame commitments across the two panels.
[Single-edit controls](evaluation-panels/video-face-second-pane-single-2026-11-14.json)

## App verification

Eight simulated prepaid checkout runs compared the follow-up off and
on. Zero jobs ran before payment, and all four paired overall results
matched. The left clip added **3.167–3.367 s** with source frames
**94, 98, 101**. The right clip added **2.133–2.333 s** with source
frames **63, 67, 70**. Each positive report checked both ranges and
all 12 source frames and crops against the original. The evidence
API, JSON, PDF and signed replay agreed. The known miss kept its
first range, and the ordinary speaker cut gained no range.
[Local app receipt](evaluation-panels/video-face-second-pane-app-2026-11-14.json)

## Try and replay

The repository default is off. To enable the trial on the local app,
set `VERITAS_VIDEO_FACE_SECOND_PANE_AVC_REVIEW=true` alongside the
existing short AVC and side-by-side flags. The source scope is upright
720p H.264 with 120 verified pictures at 30 fps. Generalization to
different people, natural calls, variable cadence, additional codecs
or edits split between sides has yet to be established.

The standard-library reader recomputes the comparison from the
published commitments. The optional FFmpeg pass checks the complete
source clock and displayed frame hashes against original deliveries.
To recreate the bank, see [the parent replay](video-face-split-track-2026-11-13.md#replay).

```sh
export PYTHONPATH=backend
receipt=docs/evaluation-panels/video-face-second-pane-2026-11-14.json
python3 -B -S -m scripts.video_face_second_pane_transfer review "$receipt"
python3 -B -S -m scripts.video_face_second_pane_transfer pixels "$receipt" --bank "$pane_bank"
backend/.venv/bin/python -B -m scripts.video_face_second_pane_transfer prepare-single "$single_bank" --bank "$pane_bank"
backend/.venv/bin/python -B -m scripts.video_face_second_pane_transfer run "$single_receipt" --bank "$single_bank" --single
python3 -B -S -m scripts.video_face_second_pane_transfer pixels "$single_receipt" --bank "$single_bank" --single
backend/.venv/bin/python -B -m scripts.video_face_moving_shortlist_app "$pane_bank" "$app_run" --second-pane
```
