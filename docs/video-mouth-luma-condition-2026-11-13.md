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

## Native readback and carrier scope

A separate FFmpeg readback reopened **110 exact native full-frame and crop
pairs** across **44 selected ranges** in **22 held clips**. The scoring
pass verified **9,152 held presentation timestamps**. The readback verified
the complete set of **2,132 rational picture timestamps** across **34 paired
H.264 4:2:0 stress deliveries**. The workspace runtime became inaccessible
before carrier scoring, so those 34 deliveries establish clock and format
parity only. Detection on the compressed carriers remains unmeasured.

The result does not support adding either arm to the app. A single qualified
MuseTalk case also cannot establish a generator-wide result. The sanitized
ledger reproduces the development points, held decisions, exact source
anchors and coverage. The raw recordings and crops stay in the local archives.

## Replay the published ledger

The standard-library reader verifies the retained scalar decisions and source
commitments without downloading the clips. Native scoring and replay require
the local input banks. [Decision ledger](evaluation-panels/video-mouth-luma-condition-2026-11-13.json)

```sh
python3 -B -IS backend/scripts/review_video_mouth_luma_native.py \
  docs/evaluation-panels/video-mouth-luma-condition-2026-11-13.json
```
