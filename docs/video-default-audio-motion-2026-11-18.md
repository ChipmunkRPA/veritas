# Mouth timing on a video’s default second soundtrack

The existing single-track mouth-timing check measures brief disagreements
between the picture and its soundtrack. A two-track container can put a
different voice or audio mix first and mark the second track as the default.
This test asks whether the existing check reproduces its source timing and
evidence when that same soundtrack occupies the default second position.
The test uses the existing comparison and its unchanged operating point.
[Single-track 44.1 kHz comparison](video-lip-motion-aac441-transfer-2026-11-18.md)

## Input scope

The test uses short upright 720 × 480 MOV videos close to 24 fps. Its
candidate requires exactly two mono 44.1 kHz AAC-LC tracks, zero starting
timestamps, and exactly one enabled audio track in the second position.
The selected track must have a verified continuous native sample clock
and fit the bounded single-shot picture timing. No more than four seconds
enter the comparison. A track’s default mark comes from the file’s track
header. It supplies no information about speaker identity. Recent
active-speaker research motivates treating a different voice as its own
control, since speech can come from someone other than the visible person.
[FFmpeg’s native track-header reading](https://ffmpeg.org/doxygen/trunk/mov_8c_source.html)
· [Active-speaker research](https://openaccess.thecvf.com/content/WACV2026/html/Nguyen_LASER_Lip_Landmark_Assisted_Speaker_Detection_for_Robustness_WACV_2026_paper.html)

The 108 carryover cases use 18 previously held recordings from nine
Lombard Grid speakers, outside the original fitting and point selection.
Each group supplies a different-word replacement, a same-word
replacement, an unchanged copy, a gain change, and two whole-track
offsets. Each nondefault track contains speech from a different
speaker in the same collection. In 18 additional controls, original
picture and synchronized first audio remain in the file while another
speaker’s recording becomes its default second track. The comparison
reads only the submitted file. The known edit interval is joined
after scoring, with 40% temporal intersection over union required for a
useful result. [Publisher data and CC BY 4.0 terms](https://zenodo.org/records/3736465)

## Results

| Measurement | Outcome |
| --- | ---: |
| Exact carryover of scores, dispositions and selected ranges | 108/108 |
| Correctly located different-word replacements | 2/18 across two speakers |
| Inherited misplaced different-word highlights | 1/18 |
| Inherited unchanged or gain highlights | 2/36 |
| Inherited same-word highlights | 2/18 |
| Inherited whole-track offset highlights | 0/36 |
| Different-speaker default controls highlighted | 1/14 measured, 4 unassessed |
| Default-track and stream-count controls withheld | 15/15 |

The original first-default, both-default and third-track controls were
withheld. The MOV writer enabled the first track when no default
designation was requested. Three separate controls cleared that native
enable bit, producing an observed `[0,0]` header. These were withheld too.
The independent reader compared every observed audio track enable flag
with the binary track header. It also verified the picture and audio packet
commitments, original presentation clocks, production decoded samples for
126 selected tracks, 24 original RGB anchors and eight selected audio
excerpts. Four different-speaker controls lacked sufficient agreement
between the duration of the default track and the picture.

This is a carrier and source-routing test on recordings already used to
check the same measurement. It establishes the same source timing after
remuxing. A spoken track from another person produced an ordinary flag,
so the report must identify the selected track. Dubbing, translation,
commentary and recording changes can affect matching. The measurement
contributes zero weight to AI confidence.

## Application and report verification

An isolated local application completed six analyses after simulated
USD 9.99 payment. It tested the same replacement with the option disabled
and enabled, the other known replacement, an ordinary no-match recording,
an ambiguous default designation, and the ordinary alternate-voice alert.
All six PDF and JSON results and signed packages verified against their
original uploads. The three highlighted excerpts reproduced the independently
decoded second-track samples. For the paired off/on test, the ordinary first
track and the overall assessment were identical. The first located interval
was 1.149–1.358 seconds, supported by source frames 26, 28 and 32.
In Chromium, all five enabled cases showed the expected timing state. The
three highlighted cases loaded their original mouth crops and played the
matching second-track excerpt at the recorded duration.

## Recount

The anonymous numeric ledger includes the input commitments, group
counts, selected times and the ordinary control outcomes. Original clips,
source identities and face crops remain local. The standard-library
reviewer checks the group boundaries, localization, prior interval
parity and frozen gate.

```sh
python3 -I -S backend/scripts/review_video_default_audio_motion.py \
  docs/evaluation-panels/video-default-audio-motion-2026-11-18.json.gz
```
