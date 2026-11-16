# Does mouth width help locate short lip-sync changes?

The local lip check compares speech with the vertical opening of the lips.
This experiment tests a complementary measurement from the same face
landmarks. The distance between the inner lip corners is projected along
the eye axis and divided by eye spacing. It can distinguish a wider mouth
shape from a rounded one at the same vertical opening. Recent research
finds that local audio and mouth behavior can reveal mismatches that broad
synchronization misses. This experiment addresses short timing shifts in
recorded speech. It reuses the existing landmark asset and Apache-2.0
speech checkpoint. [Articulation consistency research](https://arxiv.org/html/2610.30872v1)
· [Reviewed landmark asset](video-bilabial-mouth-transfer-2026-11-14.md)
· [Pinned speech checkpoint](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft/tree/ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4)

## Comparison

Recompute mouth measurements from the original frames. Require the saved
picture clock, image hashes, aperture values and crop hashes to match.
Fit a separate width predictor from grouped speech probabilities using the
same 16 unchanged recordings and four speakers as the aperture fit. Match
each predicted rank to observed ranks at candidate audio and picture lags.
The combined rule averages the two score vectors on identical sample and
picture positions. Both candidates require one timing shift with evidence
on both sides, plus original before, during and after frames.

The comparison preserves the order used in the eligible local trial.
Sparse speech events take priority, followed by relative mouth opening.
A candidate may supply an interval only when both existing checks leave
the clip without one. The tight points for the incumbent equal the local
settings. The tight and wider candidate points use the largest and
second-largest per-speaker maxima from four other speakers, using ordinary
original, gain and filter controls plus uniform soundtrack shifts. The
candidate points stay fixed on every test panel. A correct result overlaps
the known interval by at least 0.40 temporal intersection over union and
gets the signed timing shift within 80 ms. [Existing mouth-opening trial](video-lip-aperture-scale-2026-11-15.md)

## Results

The 64 main recordings have 16 held speakers. The two delivery tests use
the same 16 recordings from eight separate speakers, first as PCM and then
as 48 kbps AAC at 16 kHz. The actor recordings use 45 additional speakers
at native 29.97 fps across two frozen source groups. Their soundtrack edits
use the same paired controls and the original frame clocks. The actor
comparison is an offline test of the rules beyond the current lip-timing
frame-rate coverage. All panels reuse retained sources with earlier
project exposure. [Lombard Grid recordings and CC BY 4.0 terms](https://zenodo.org/records/3736465)
· [Actor source qualification and terms](video-lip-frame-clock-2026-11-15.md#separately-recorded-2997-fps-speech)

| Test panel, tight point | Correct edit ranges, existing → combined fallback | Ordinary controls flagged | Uniform shifts flagged as local |
| --- | ---: | ---: | ---: |
| Main PCM | 18/128 → 21/128 | 2/192 → 5/192 | 5/128 → 6/128 |
| Separate speakers, PCM | 5/32 → 5/32 | 0/48 → 0/48 | 3/32 → 3/32 |
| Same separate speakers, AAC | 4/32 → 5/32 | 0/48 → 0/48 | 1/32 → 2/32 |
| First 24 actors | 5/48 → 6/48 | 5/72 → 5/72 | 2/48 → 2/48 |
| Remaining 21 actors | 4/42 → 4/42 | 0/63 → 3/63 | 1/42 → 1/42 |

The combined fallback found three additional main PCM ranges from two
speakers and misplaced another three. Unassessed main edits fell from
8/128 to 0/128. One additional AAC result flagged
**0.440–1.160 s** with a **200 ms** estimated delay. The report binds
that proposal to original picture indices **5, 14 and 24**, their
timestamps and the original crops. The same rule added one correct range
in the first actor panel and a misplaced range in the second. In the main
panel, the width-only fallback raised correct ranges to 20/128 and uniform
shift flags to 10/128. At the wider point, the combined fallback added
three correct main locations and one actor location, with six added
misplaced locations across those panels. The total misses and unassessed
cases can be derived from each receipt's `files`, `useful` and `assessed`
counts. [Grid numeric and frame receipt](evaluation-panels/video-lip-mouth-width-grid-2026-11-15.json.gz)
· [Native actor aggregate](evaluation-panels/video-lip-mouth-width-native-2026-11-15.json)

The width checks stay in evaluation. The gains were small and uneven
across speakers and deliveries. The stage that adds a useful AAC location
also adds one constant-delay alert, and the second actor panel gains
ordinary alerts without a correctly placed edit. The comparison measured
timing on staged edits. A generated or manually retimed mouth may still
match the soundtrack closely. Timing evidence alone leaves the type of
edit unresolved.

## Replay

The portable command recomputes the operating points, attempted coverage,
false alarms, ranges and picture witnesses from the public numeric Grid
receipt. It gives the same output on Python 3.12 and 3.14 using only the
standard library. The two actor panels are published as aggregate counts.
Repeating the source decode reproduced all 112 Grid mouth-width series
and both actor aggregates byte for byte.
Original clips, full speech probabilities and individual actor records
stay local.

```sh
export PYTHONPATH=backend
module=scripts.video_lip_mouth_width_transfer
grid=docs/evaluation-panels/video-lip-mouth-width-grid-2026-11-15.json.gz
python3 -B -S -m "$module" review "$grid"
```

Native replay needs the two Grid archives and the prior
[native qualification inputs](video-lip-frame-clock-2026-11-15.md#replay).
Use the same Mac GPU profile for acoustic inference to reproduce the
historical aperture predictions exactly. A CPU comparison decoded the
same audio samples but produced different speech estimates. The two
candidate checks reuse the same acoustic estimates as the incumbent.
The first command below rebuilds the earlier full speech-probability
cache locally. The native transfer recomputes both actor panels from
qualified media at their original timestamps.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1
assets=models/research/video-lip-timing
runner=backend/.venv/bin/python
"$runner" -B -m scripts.video_lip_dense_posteriors capture "$run/primary.jsonl" \
  --checkpoint "$assets/checkpoint" --sources "$archive"
"$runner" -B -m "$module" pictures "$archive" "$assets/face_landmarker.task" "$assets/runtime" "$run/widths.json.gz"
"$runner" -B -m "$module" carrier-sound "$archive" "$assets/checkpoint" "$run/carrier.jsonl"
"$runner" -B -m "$module" report "$run/widths.json.gz" "$run/primary.jsonl" "$run/carrier.jsonl" "$run/grid.json.gz"
cmp "$grid" "$run/grid.json.gz"
"$runner" -B -m "$module" native-transfer "$run/grid.json.gz" "$audit_dir" "$publisher_media_dir" \
  "$assets/checkpoint" "$assets/face_landmarker.task" "$assets/runtime" "$run/native.json"
cmp docs/evaluation-panels/video-lip-mouth-width-native-2026-11-15.json "$run/native.json"
```
