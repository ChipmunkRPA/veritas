# Native chroma sampling phase across short mouth edits

This single-file experiment checks whether the color channels in a short mouth
redraw lose a repeating pattern tied to the surrounding picture's decoded pixel
grid. An edited mouth may have passed through a different color and resampling
pipeline even when it follows the original speech. Ordinary resampling can
break the same pattern. The measurement establishes neither camera identity nor
the process that caused the difference. Earlier image forensics research
demonstrates that chroma subsampling can leave position-specific artifacts.
Recent lip-region research motivates looking locally around the mouth. This
study uses no external detector code or weights beyond the previously reviewed
face geometry asset. [Chroma subsampling research](https://cris.fau.de/publications/217904346/) ·
[Lip-region frequency research](https://arxiv.org/abs/2609.22284) ·
[Face geometry license](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

## Frozen method

Each decoded native RGB picture supplies two color-difference channels. Native
pixel pairs are grouped by the parity of their horizontal or vertical
coordinate. Eye and mouth landmarks determine whether each pair lies in the
mouth ellipse, a cheek patch or the upper face. The detector never transforms
the native sampling grid. Strong brightness edges and saturated pixels are
excluded. Pairs at both phases and at least two surrounding patches must show
enough signal. Each shot learns its reference phase and sign from the cheek and
upper face. The first arm uses its strongest valid axis. The second arm requires
support on both axes and keeps the weaker of their observed losses. Shots with
unstable or weak surrounding phase provide no measurement.

Each arm searches 8-, 12-, 16- and 20-picture ranges, each bracketed by five
native pictures from the same shot. A reported interval must show a consistent
loss in the middle relative to both neighboring ranges. A deterministic array
test recovered a known interval in which both color phases were replaced and
returned insufficient support after the entire synthetic picture lost its
pattern. The same test does not establish performance on real clips.

The third-largest ordinary maximum across 56 GRID development deliveries sets
the decision point for each arm. The fixed points were 0.358086 and 0.446596.
The 146 held inputs contain 23 neural edits and 123 ordinary files. Ten
LivePortrait and one MuseTalk edit retained an independently measured local
speech event. The 123 controls include untouched recordings, genuine cuts,
ordinary resampling and retiming, conventional local temporal processing and
re-encodes. The same media had prior exposure under other project methods.
The new carrier stress separately encodes all 17 source-time redraws and
their originals using H.264 CRF 23 with 4:2:0 chroma. Each transcode retains
the exact rational picture clock and omits the soundtrack. The detector itself
uses only the submitted pictures. All labels and authored ranges enter the
evaluation after scoring. [Prior held roster and ordinary derivatives](video-mouth-track-cycle-2026-11-13.md) ·
[New second-generator source audit](lombard-fresh-musetalk-2026-11-13.md) ·
[GRID terms](https://zenodo.org/records/3625687) ·
[Lombard terms](https://zenodo.org/records/3736465)

## Results

| Inputs | Dominant-axis coverage | Both-axes coverage | Locations and flags |
| --- | ---: | ---: | --- |
| Held neural edits | 2/23 | 0/23 | 0/23 localized by either arm |
| Held ordinary controls | 28/123 | 12/123 | 0/123 flagged by either arm |
| H.264 4:2:0 neural edits | 2/17 | 0/17 | 0/17 localized by either arm |
| H.264 4:2:0 ordinary controls | 1/17 | 0/17 | 0/17 flagged by either arm |

Both arms measured all 56 development clips. In the primary held bank, the
dominant-axis arm measured two of ten strict LivePortrait edits and none of
the independently qualified MuseTalk edit. The both-axes arm measured none
of those 11 cases. One dominant-axis proposal covered **0.414–1.076 s** with
0.778 temporal intersection over union against the authored **0.331–0.993 s**
range. Its score was 0.30953, below the fixed 0.35809 point. The other measured
neural file had no positive proposal. The high number of unassessed cases
limits what can be inferred from the absence of ordinary flags.

The native replay verified the three full pictures and mouth crops supporting
the below-point proposal. It also independently verified the H.264 4:2:0
pixel format and all 2,132 stress timestamps. The camera sources used in point selection had
a stable phase on both axes. Most held clips, including the recompressed clips,
showed a consistent phase on only one axis or had insufficient reference
phase entirely. The trial yielded no support for adding either check to the
app or its confidence estimate. [Decision and source exhibit ledger](evaluation-panels/video-mouth-chroma-grid-2026-11-13.json)

## Replay

Use the retained native video banks and the earlier trajectory study's control
directory. Source videos and native crops stay local. The public ledger stores
the decisions and clock measurements. The separate native reader reopens the
before, during and after native pictures and exact crops for selected
proposals, and compares every stress carrier clock against its original.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for phase in freeze prepare development held codec_stress; do
  backend/.venv/bin/python -B -m scripts.video_mouth_chroma_grid "$phase" \
    "$run" "$lombard" "$fresh" "$musetalk" "$trajectory_controls" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_mouth_chroma_grid "$receipt" --assemble "$run"
python3 -B -IS backend/scripts/review_video_mouth_chroma_grid.py "$receipt"
```
