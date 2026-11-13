# Repeated speech and signed mouth motion

A single-upload check compared the changes in two sequences of mouth
pictures when the soundtrack repeats. On the source-disjoint CREMA-D
transfer, it located the same **16 of 40** known soundtrack edits as the
previous static-mouth comparison in both PCM and AAC. Both checks also
flagged the same synchronized mouth-and-sound copy, **one of 90** ordinary
controls per carrier. The release timing check assessed none of these
2.52-second clips. A separate picture-context guard added three useful
pairs when combined with the static check. The experimental checks remain
outside the app.

## What was measured

The soundtrack comparison nominates two non-overlapping spans with at least
240 ms of repeated active speech. Exact timestamps pair mouth pictures in
the two spans. The static comparison looks for different pictures despite
the similar audio. The signed-motion comparison uses changes between
consecutive eye-aligned mouth pictures. It looks for two active sequences
whose changes disagree under one common small spatial translation. That
translation can account for crop placement across the sequence. The check
also requires a comparable observed head pose and rejects repeated wide
pictures and similar native mouth patches.

Recent work separates speech timing from visible articulation and finds
that the two can disagree. This trial uses that distinction to ask whether
similar recorded audio corresponds to similar observed mouth motion. It
uses no external model, weights, or source clips from those studies. It
uses the existing MIT-licensed YuNet face geometry checkpoint and imports
no speech-recognition weights. [PVSync](https://arxiv.org/abs/2610.09223) ·
[AV-SyncBench](https://arxiv.org/abs/2607.00726) ·
[Interpretable lip timing](https://arxiv.org/abs/2609.06788) ·
[Pinned geometry license](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

Four actors set the internal point at 0.65 from four predeclared distances.
The choice maximized useful development pairs without flagging an ordinary
development file. Eight qualified actors from the prior repeat-speech trial
are a retained comparison. Sixteen different planned actors come from two
more recent paired picture studies. Ten of those actors supplied a common
63-picture and 40,320-sample extent. The other six remained recorded
source-duration declines before scoring. All source groups were disjoint
between development, retained and transfer. The source recordings had prior
project exposure. The transfer soundtrack edits are newly authored from
archived native sources. Each recording contributed four known soundtrack
edits and nine ordinary controls. The carriers share the same actors and
edits. [Original repeat method and controls](video-lipsync-repeat-audio-2026-11-12.md) ·
[Archived transfer sources](video-reference-measured-interval-2026-11-13.md) ·
[Pinned publisher terms](https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt)

A location needs at least half temporal overlap in both authored repeat
witness ranges. The transfer rule requires six additions across at least
four actors while preserving earlier locations. It allows one additional
ordinary alert and zero synchronized-copy or whole-track-offset alerts.
The evidence cannot identify which of the two positions was edited. The
panel tests ordinary soundtrack shifts or copies and gives no performance
estimate for synchronized neural redraws.

## Results

Each count below is per carrier, with identical decisions in PCM and AAC
on the retained and transfer cases. Acoustic-only results are nominations.
The current timing check returned no assessed short windows in these views.

| Panel and result | Acoustic nominations | Static guard | Signed motion | Union of both checks |
| --- | ---: | ---: | ---: | ---: |
| Retained useful locations | 17/32 | 3/32 | 3/32 | 3/32 |
| Retained ordinary flags | 16/72 | 0/72 | 0/72 | 0/72 |
| Transfer useful locations | 27/40 | 16/40 | 16/40 | 16/40 |
| Transfer ordinary flags | 21/90 | 1/90 | 1/90 | 1/90 |
| Transfer synchronized or global controls with a flag | 18/30 | 1/30 | 1/30 | 1/30 |

One transfer delivery with a known earlier local soundtrack move had exact
repeat witnesses at **1.440–1.760 s** and **1.760–2.080 s**. The selected
mouth pairs came from frame indices **36–43** and **44–51**, respectively.
The evaluation receipt carries exact PCM sample bounds and SHA-256 commitments
for the native observations. The selected mouth rectangles indicate where
the measurements came from. A separate FFprobe and FFmpeg pass verified 3,276
native picture timestamps, 2,113,280 decoded samples, 114 exact PCM slices
and 910 native mouth crops across 52 selected deliveries.

## Guard using the surrounding picture

A diagnostic on the previously exposed eight-actor repeat bank found that
18 of 29 acoustic nominations on affected PCM clips failed the fixed
whole-picture difference floor. That failure can occur on genuine, mostly
static shots. A follow-up therefore tested whether the non-mouth pixels
show a localized difference in two picture quadrants, outside the two
measured mouth regions. It also required strong signed-motion disagreement
if the earlier static-mouth check considered the two patches similar. The
follow-up reused the same actor banks and edit construction. The operating
point remained 0.65 under the same development rule.

| Panel and result | Previous static guard | New picture-context guard | Union of both checks |
| --- | ---: | ---: | ---: |
| Retained useful locations | 3/32 | 4/32 | 4/32 |
| Retained ordinary flags | 0/72 | 0/72 | 0/72 |
| Transfer useful locations | 16/40 | 16/40 | 19/40 |
| Transfer ordinary flags | 1/90 | 1/90 | 1/90 |

The three added transfer locations came from three actors, with no new
ordinary flags and no lost static locations in the union. Each individual
check missed three edit pairs that the other found. Both still flagged the
same synchronized mouth-and-sound copy. The two carriers produced the same
decisions. The follow-up missed the frozen six-gain, four-actor and
synchronized-control requirements. Both the retained and transfer groups
had already been measured by the first trial. A newly sourced test should
verify the apparent complementarity and measure synchronized copy errors
before use. These outcomes do not establish detection of new neural mouth
pictures that preserve timing.

One additional transfer clip selected **0.540–0.990 s** and
**1.580–2.030 s**. The authored repeat witnesses were **0.560–1.040 s** and
**1.600–2.080 s**. The selected native pictures were indices **14–23** and
**40–49** on a 25 fps timeline. [Original motion ledger](evaluation-panels/video-lipsync-repeat-motion-2026-11-13.json) ·
[Picture-context ledger](evaluation-panels/video-lipsync-repeat-picture-guard-2026-11-13.json)

## Replay

Supply the retained prior repeat run, its source intake, and the two earlier
paired-picture source freezes. The native worker reconstructs the picture
and audio carriers, checks unchanged picture clocks, and separately reopens
flagged frame and PCM commitments using FFprobe and FFmpeg. Keep native
media and the source-to-actor map local.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze development calibrate retained transfer report; do
  backend/.venv/bin/python -B -m scripts.video_lipsync_repeat_motion \
    "$phase" "$run" "$previous" "$sources" "$picture_development_freeze" "$picture_held_freeze" || exit 1
done
python3 -B -IS backend/scripts/review_video_lipsync_repeat_motion.py "$run/receipt.json"
```

Repeat the same phases with `scripts.video_lipsync_repeat_picture_guard`
and a new run directory to reproduce the follow-up.
