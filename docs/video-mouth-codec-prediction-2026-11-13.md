# H.264 inter-picture prediction around short mouth redraws

This single-file test measures the submitted encoder's block predictions around
the mouth. Compressed-domain forensic research uses motion vectors and
prediction footprints to study video manipulation and encoding history. This
experiment uses FFmpeg's exported decoder motion vectors and the existing
MIT-licensed face geometry model. The research detector code and weights are
outside the experiment. [Motion-vector localization research](https://doi.org/10.1016/J.COMPELECENG.2020.106929) ·
[H.264 forensic feature research](https://link.springer.com/article/10.1186/s13635-024-00181-4) ·
[Face geometry licence](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

## Frozen comparison

The local decoder extracts P-picture vector rectangles and verifies their
presentation timestamps against the decoded picture stream. Face geometry
positions a mouth ellipse and three comparison patches. The coverage arm
compares how much of the mouth and nearby face is covered by exported vectors.
The dispersion arm compares how far the local vectors deviate from the
surrounding region's median motion vector after normalizing face scale. The
measurements describe encoded inter-prediction decisions. Texture, motion,
occlusion, and encoder settings can change both.

Each arm keeps its strongest 8-, 12-, 16- or 20-picture range, with measured
five-picture flanks on both sides in one observed shot. The third-largest
ordinary maximum among 56 GRID development deliveries set each point, with a
floor of 0.04 and strict exceedance. A correct location overlaps at least half
of the known authored interval by temporal intersection over union.

The held panel has 17 earlier source-time redraws from two neural workflows,
their 17 original counterparts, 34 source-matched ordinary resampling and
retiming edits, and 24 additional ordinary processing, encoding and genuine-cut
controls. Ten LivePortrait cases and one MuseTalk case had separately verified
local speech events. The source-time edit truth and speech-event labels enter
evaluation after scoring. These are retained sources with prior project
exposure. All deliveries use H.264 CRF 23, 4:2:0 chroma, and no B pictures.
The 34 paired carriers were retained from the earlier brightness-conditioned
test. The other 114 were encoded using the same settings with exact rational
picture clocks. This tests one encoding profile and short prepared edits.
[Prior source and terms](video-mouth-track-cycle-2026-11-13.md) ·
[Independent second-generator qualification](lombard-fresh-musetalk-2026-11-13.md) ·
[Existing carrier study](video-mouth-luma-condition-2026-11-13.md)

## Held results

| Outcome | Inter-prediction coverage loss | Motion-vector dispersion |
| --- | ---: | ---: |
| Measured edits | 17/17 | 17/17 |
| Measured ordinary clips | 75/75 | 75/75 |
| Correct edit locations | 5/17 | 0/17 |
| Correct locations on verified LivePortrait cases | 1/10 | 0/10 |
| Correct location on verified MuseTalk case | 1/1 | 0/1 |
| Flagged edits with a misplaced range | 8/17 | 0/17 |
| Ordinary clips flagged | 37/75 | 4/75 |
| Original clips flagged | 12/17 | 1/17 |

The 56 development files were measured by both arms. The fixed operating
points were 0.057602 for coverage loss and 0.415756 for dispersion. The
coverage arm flagged 23 of 34 matched ordinary edits and two of 12
additional local temporal-processing controls. Neither arm flagged any
of the 12 separate re-encode and cut controls. The earlier luminance
detail check found zero edits and flagged zero originals on these same 34
paired carrier files. [Paired luminance comparison](video-mouth-luma-condition-2026-11-13.md#h264-carrier-results)

The verified MuseTalk case had an authored range of **300/289–492/289 s**.
The coverage arm selected **312/289–408/289 s**, with temporal IoU 0.500
and a score of 0.072230. Before, during and after source frames were
**25, 30 and 34**, at **300/289, 360/289 and 408/289 s**. The separate
LivePortrait result also had temporal IoU 0.500. Both cases meet the
overlap rule at its boundary. A source frame shows the observed mouth area.
The evaluation's authored range establishes where the experimental edit
occurred.

Coverage loss found short neural redraws that the earlier luminance cue
missed, while triggering on most originals from the paired bank. Its
maximum was higher in the redraw than in the corresponding original in
only six of 17 pairs. Dispersion's maximum was higher in 12 of 17 pairs,
but it produced no qualified locations at its frozen point. This narrow
one-profile experiment does not support using either arm in the app.

An independent FFmpeg pass reopened all **4,200 development** and **5,834
held** presentation timestamps against the originals. It also verified
**188 exact frame and crop pairs** across **64 proposed ranges** in **54
held files**. The isolated public replay checked the same selected
commitments and decision totals.

## Local replay

The isolated standard-library reader recalculates points, decisions and
overlap from the sanitized ledger. Reopening media requires the retained
originals and local FFmpeg. The experiment itself also needs FFmpeg development
headers and the repository's Python environment. The raw frames, face geometry
and vectors stay local. [Decision ledger](evaluation-panels/video-mouth-codec-prediction-2026-11-13.json)

```sh
clang -std=c17 -Wall -Wextra -Werror $(pkg-config --cflags libavformat libavcodec libavutil) \
  backend/scripts/read_video_motion_vectors.c $(pkg-config --libs libavformat libavcodec libavutil) -o "$exporter"
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze prepare development held; do
  backend/.venv/bin/python -B -m scripts.video_mouth_codec_prediction "$phase" "$run" "$prior" "$exporter" || exit 1
done
python3 -B -IS backend/scripts/review_video_mouth_codec_prediction.py "$receipt" --assemble "$run"
python3 -B -IS backend/scripts/review_video_mouth_codec_prediction.py "$receipt"
```
