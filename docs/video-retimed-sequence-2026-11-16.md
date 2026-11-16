# Locate footage reused at two playback speeds

Chipmunk Verify can show two portions of an upload where similar moving
pictures appear at twice the playback speed in one portion. Each range
links to its first and last decoded source pictures and playback position.
The same ranges and four pictures appear in the standard PDF, JSON and
signed evidence package. Changing playback speed is common in ordinary
editing. The observation carries zero AI-confidence weight.

The check compares one native frame step against two steps on the same
verified presentation clock. It uses the existing whole-picture luminance
descriptor and compares both appearance and the direction of the changes
between paired pictures. It requires enough changing pictures to support
the match, two disjoint ranges, and at least half a second between the
matched endpoints of the shorter range. This is a separate comparison
from the equal-speed and reverse-playback checks, which have priority.
The method builds on research about how time changes affect copy
detection. [Temporal changes in video copy detection](https://arxiv.org/abs/2501.11171)

## Local results

Four retained source recordings supplied the development cases. Another
eight captures supplied the held test. Two of the eight were new to the
project and six came from earlier camera and repeat evaluations. Source
selection, comparison rules and evaluation criteria were committed to
the local run record before opening the held outcomes. Eight captures
supplied four retimed edits each, with the faster and slower occurrence
placed in both possible orders. Ordinary controls included the original,
an in-place speed change, a jump cut, a held picture and a recompressed
original. Same-speed and reverse repeats supplied separate controls for
preserving the existing readings.

| Held result | Existing repeat checks | With the retimed comparison |
| --- | ---: | ---: |
| Retimed copies with both ranges and their relative speed located | 0/32 | 26/32 |
| Both authored ranges recovered exactly | 0/32 | 19/32 |
| Retimed copies located after a second, more aggressive encode | 0/8 | 0/8 |
| Ordinary clips with a repeat alert | 5/40 | 5/40 |
| Known same-speed and reverse repeats retained | 16/16 | 16/16 |

A useful result needs at least 50% intersection over union on each
authored frame range and the correct relative speed. The new comparison
added no alerts across the 56 controls. The existing ordinary alerts
all came from a sprint recording with recurring pictures. Those existing
matches also took priority for that recording's four staged retimed
variants. On the two newly sourced captures, the check found all eight
single-encode edits. Across all eight captures it found a useful edit in
seven. The controlled panel covers frame skipping and duplication at a
fixed ratio within one upload. Heavily compressed footage and other
speeds, cropping and interpolated motion need separate validation.

The local option, `VERITAS_VIDEO_RETIMED_SEQUENCE_REVIEW`, checks the
first 225 unrotated 640 × 360 frames of 25 fps H.264 MP4 in 8-bit 4:2:0
with verified, regular source timestamps. The option defaults off.
Test media was decoded and re-encoded locally. The ledger contains
source credits and measurements. Raw source and altered media stay local.
[Held case ledger](evaluation-panels/video-retimed-sequence-2026-11-16.json)
· [VCD2 data terms](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA)

## Local app verification

The ordinary view showed both ranges, their relative speed, the four
original frames and a playback link for each frame. The first example
located **1.160–2.760 s** and **5.680–6.480 s** on the source clock. The
browser sought to **5.680 s** when the start of the second range was
selected. The API, JSON and PDF supplied the same ranges, and the four
original images matched those embedded in the PDF. Signed export
verification recomputed both matches from the submitted videos. Five
simulated USD 9.99 purchases queued no jobs before payment. The positive
replayed with the option off and on also produced the same AI assessment.
[Local application receipt](evaluation-panels/video-retimed-sequence-app-2026-11-16.json)

## Reproduce

The portable review independently recounts interval overlap, exact
boundaries, frame strides and alert counts. Use the optional media path
to verify the endpoints with separate FFprobe timestamp and FFmpeg RGB
frame-hash reads. The local replay checked all 104 reported endpoints
across 5,850 native decoded pictures. The arithmetic recount matched
on Python 3.12 and 3.14.

```sh
export PYTHONPATH=backend
receipt=docs/evaluation-panels/video-retimed-sequence-2026-11-16.json
python3 -B -S -m scripts.review_video_picture_speed_panel "$receipt"
```

For a native replay, copy the receipt's `sources` list into a local JSON
file and add a `path` to each original downloaded or restored from the
publisher link. The builder checks the matching source hashes and
establishes all delivery hashes and edit ranges before scoring. Use a
fresh run directory and the project's Python environment with FFmpeg.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_picture_speed_panel build "$run" "$sources"
backend/.venv/bin/python -B -m scripts.video_picture_speed_panel score "$run" "$run/result.json"
python3 -B -S -m scripts.review_video_picture_speed_panel "$run/result.json" --media "$run"
```
