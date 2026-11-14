# Learning short audio and mouth correspondences

Chipmunk Verify's existing timing check uses mouth opening and audio energy.
This comparison trained a small model to distinguish corresponding audio
and mouth observations from nearby mismatches in the same recording. This
holds the person, recording conditions and background as constant as
possible during training. Recent work uses matching and mismatched inputs
to learn audiovisual timing and separately evaluates articulation. This
experiment tests timing and uses no checkpoints or code from those papers.
[Timing and articulation](https://arxiv.org/abs/2610.09223)
· [Audiovisual synchronization in the wild](https://arxiv.org/abs/2112.04432)

## Comparison

The fit uses 28 newly prepared excerpts from seven Wikitongues recordings.
It keeps 879 same-time samples with detected speech and a measurable mouth,
plus six mismatches per sample at offsets of ±160, ±240 and ±360 ms. Each
sample spans five 40 ms ticks. Mouth inputs are the existing opening proxy
and coarse gray image cells. Audio inputs are log energy and relative band
powers. The fit uses 12 whitened components from each modality and a
regularized logistic model on their pairwise products. The existing face
detector supplies mouth locations. It is the only pre-trained weight used.
[Face weight terms](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

A proposal tests which overlap of mouth and sound is favored in a short
interval and in two surrounding flanks. Both flanks must prefer the same
lag and the middle must prefer a different lag on the same mouth samples.
The decision point of 2.061890 is 0.03 above the largest ordinary response
on eight excerpts from four different speakers. The held panel then uses
two new excerpts from each of five further recordings. These speakers were
outside fitting and point selection, and their source files had prior
project use. The controls include unchanged recordings, gain, filtering,
local noise, constant whole-track shifts and picture observations moved
with the audio. The synchronized controls move the measured picture inputs
on the same grid. The study does not re-encode those controls.
[Case construction](video-lipsync-signed-transitions-2026-11-14.md)
· [Source credits, terms and per-excerpt readout](evaluation-panels/video-local-alignment-2026-11-14.json)

## Held results

| Outcome | Current check | Learned correspondence |
| --- | ---: | ---: |
| Staged edits with a timing measurement | 7/20 | 6/20 |
| Ordinary controls with a timing measurement | 24/80 | 27/80 |
| Useful locations at the frozen point | 0/20 | 0/20 |
| Controls flagged | 0/80 | 0/80 |

A useful location needs at least 0.40 overlap by temporal intersection
over union and a signed lag within 80 ms of the authored 200 or 280 ms
shift. Two planned excerpts lacked a measurable mouth on all their ticks,
and remain in the denominator. One candidate before applying the decision
point had the right direction, acceptable lag and overlap. Its score
was smaller than the strongest ordinary response on the same held panel.
The learned method missed the advancement rule and remains offline.
These are staged short timing shifts on genuine clips. The experiment
does not measure AI origin or the articulation of synchronized redraws.

## Source clock evidence in the app

When the decoded audio clock prevents lip-sync assessment, new reports show
the previous span's end, the next span's start, the boundary sample and the
direction of the jump. The analysis and Standard PDF show the first eight
breaks. The JSON includes all breaks with exact source ticks and their time
base. The signed verifier checks the listed breaks against the recorded
timeline and can recompute them from the submitted file. An encoded test
exercised two forward breaks, and a browser fixture checked both directions.
These timestamps explain why lip sync was unassessed. They do not affect AI
confidence.

## Replay

The standard-library reader rechecks the source roles, edit clocks,
operating point and error counts. Two native runs produced identical
receipt bytes. Use the exact publisher files in `freeze.sources`.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
receipt=docs/evaluation-panels/video-local-alignment-2026-11-14.json
python3 -B -S -m scripts.video_local_alignment_transfer review "$receipt"
backend/.venv/bin/python -B -m scripts.video_local_alignment_transfer run "$run" \
  --cache "$prior_conversations" --held "$separate_conversations"
cmp "$receipt" "$run/receipt.json"
```
