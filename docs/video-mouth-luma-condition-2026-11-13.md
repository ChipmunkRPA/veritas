# Brightness-conditioned picture detail on short mouth redraws

Image splicing studies compare visual noise at similar brightness levels. This
single-upload test asks whether a short mouth redraw changes fine luminance
detail at similar brightness and coarse gradients. The measurement uses
decoded pixels, which also reflect ordinary texture, resizing, denoising and
compression. The only pretrained asset is the existing MIT-licensed face
geometry. [Signal-dependent noise research](https://arxiv.org/abs/2010.16211) ·
[Spatial and temporal video residuals](https://arxiv.org/html/2503.20991v1) ·
[Face geometry licence](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

## Method

The check measures the diagonal Haar-band detail of native four-by-four
luminance blocks. Eye and mouth landmarks select the mouth area and three
surrounding comparison areas. The plain arm compares median fine detail across
those regions. The brightness-conditioned arm matches each mouth block to
three surrounding blocks within fixed luminance and coarse-gradient bounds.
Both scan short ranges with five-picture flanks in one observed shot and
retain the strongest supported drop against both flanks. A deterministic
array check passed local detail suppression, a global luminance offset and
recovery of a known tapered change.

The third-largest ordinary maximum in 56 GRID development files set each
point, subject to a fixed floor. The held files come from previously qualified
sets with known edit ranges and separate speech-event observations. They
include 23 neural deliveries and 123 untouched or ordinarily processed files.
The detector uses only the submitted pictures. The evaluation joins the
known edit ranges after scoring. A correct range requires temporal IoU of at
least 0.50. The clips had earlier project exposure. [Prior roster and source terms](video-mouth-track-cycle-2026-11-13.md) ·
[Second-generator speech-event qualification](lombard-fresh-musetalk-2026-11-13.md)

## Held native results

| Outcome | Plain detail | Brightness-conditioned detail |
| --- | ---: | ---: |
| Measured neural edits | 23/23 | 23/23 |
| Measured ordinary files | 123/123 | 123/123 |
| Correct neural locations | 0/23 | 0/23 |
| Correct source-time LivePortrait locations | 0/10 | 0/10 |
| Correct independently qualified MuseTalk location | 0/1 | 0/1 |
| Ordinary flags | 0/123 | 0/123 |

Both arms measured all 56 development files. Their frozen points were
0.385164 for plain detail and 0.374413 for conditioned detail. The strongest
ordinary response exceeded the strongest independently verified neural response
in each arm. The plain arm's highest ordinary response came from a genuine hard
cut. The conditioned arm's highest came from a routine re-encode.

The independently qualified MuseTalk edit spanned **1.038–1.702 s**. The
conditioned arm proposed **1.038–1.869 s** with temporal IoU 0.800 and a score
of 0.134478. This proposal fell below the frozen point. Both arms proposed
ranges that met the overlap criterion on 22 of 23 neural files before applying
the points. The edited ranges sit near the center of these short prepared
clips. Overlap in this panel alone gives weak evidence that the method would
locate edits with arbitrary positions. The released short-clip lip-timing check
previously had insufficient coverage for all 44 files in the LivePortrait
subset. This trial measured that subset but yielded no qualified addition.
[Released timing comparison](lombard-neural-mouth-redraw-2026-11-12.md)

## H.264 carrier results

The follow-up scored the 17 source-time redraws and their 17 matched originals
after H.264 CRF 23 encoding with 4:2:0 chroma. It used the same scoring code
and both original development points. Both arms measured all 34 deliveries
and all 2,132 pictures. Neither localized any of the 17 edits or flagged any
of the 17 originals. Before the operating points, 13 of 17 plain proposals
and 12 of 17 conditioned proposals reached temporal IoU 0.50. The clips
retain the same prepared edit positions as the native panel. [Carrier construction](video-mouth-chroma-grid-2026-11-13.md)

The conditioned proposal for the independently qualified MuseTalk case stayed
at **300/289–540/289 s** after recompression, with temporal IoU 0.800 against
the authored **300/289–492/289 s** range. Its score fell from 0.134478 to
0.022784. Both scores were below the frozen point of 0.374413. The before,
during and after crops reopened at pictures **24, 35 and 45**, corresponding
to **288/289, 420/289 and 540/289 s**.

## Native readback

A separate FFmpeg readback reopened **110 exact native full-frame and crop
pairs** across **44 selected ranges** in **22 held clips**. The scoring
pass verified **9,152 held presentation timestamps**. The readback also
verified all **2,132 rational picture timestamps** against the originals
across the **34 paired H.264 4:2:0 stress deliveries**. It reopened **66
H.264 full-frame and crop pairs** across **25 selected ranges** in **14 clips**.

The result does not support adding either arm to the app. A single qualified
MuseTalk case also cannot establish a generator-wide result. The sanitized
ledger reproduces the development points, held decisions, exact source
anchors and coverage. The raw recordings and crops stay in the local archives.

## Replay the published ledger

The standard-library readers verify the retained scalar decisions and source
commitments without downloading the clips. The initial ledger remains frozen
at the native pass. The follow-up ledger records the completed carrier pass
alongside the same development and native decisions. Reopening the original
pixels requires the retained local input banks. [Native decision ledger](evaluation-panels/video-mouth-luma-condition-2026-11-13.json) ·
[Completed carrier ledger](evaluation-panels/video-mouth-luma-condition-h264-2026-11-13.json)

```sh
python3 -B -IS backend/scripts/review_video_mouth_luma_native.py \
  docs/evaluation-panels/video-mouth-luma-condition-2026-11-13.json
python3 -B -IS backend/scripts/review_video_mouth_luma_condition.py \
  docs/evaluation-panels/video-mouth-luma-condition-h264-2026-11-13.json
```

To repeat the complete local test, use the retained chroma-grid run as
`prior` and a new private output directory as `run`.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze development held codec_stress; do
  backend/.venv/bin/python -B -m scripts.video_mouth_luma_condition "$phase" "$run" "$prior" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_mouth_luma_condition "$receipt" --assemble "$run"
```
