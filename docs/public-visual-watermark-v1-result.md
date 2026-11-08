# Public visual watermark localization runs locally; real-media qualification remains

The [publicly frozen contract](public-visual-watermark-v1-protocol.md) completed
**72 image cases,96 video-frame cases,18 generator and173 detector forwards**.
Code/assets/readout were published at
`8a4aad3310b5c37d0d44d6ce799bb5ff058c1376` before model construction or inference.
The separate [aggregate and receipts](evaluation-panels/public-visual-watermark-v1-result.json)
retain false responses, misses and overlap, without a point change or omitted case.

**Implementation evidence only; research remains outside serving, confidence,
provider attribution and early stopping.** These are six non-neural authored
carriers and three derived moving clips, not real photographs, genuine provider
outputs, face swaps or a population accuracy sample. A compatible public mark
can be added to genuine material. It is not an AI-edit or authorship proof.

## Spatial recovery and ordinary-edit controls

Partial marks occupy the center128x128 region at x64,y64 in256x256 RGB. Every
other input pixel remains identical before codec/resize transforms. Ordinary-edit
controls rotate the same center crop and receive no mark. The fixed native
presence-logit readout retains all pixels; inspection requires an8-connected
component of at least500 pixels. That proposed inspection rule is not a
calibrated false-alarm or AI-confidence guarantee.

Each image cell below has six base carriers. IoU compares selected pixels to
the intended marked region;1 means perfect overlap. Counts across transforms
reuse the same sources and are not independent population observations.

| Input transform | No-added-mark carriers flagged | Ordinary edits flagged | Full marks missed | Partial marks missed | Mean partial IoU |
|---|---:|---:|---:|---:|---:|
| Native RGB | 0/6 | 0/6 | 0/6 | 0/6 | .9661 |
| JPEGquality70 | 1/6 | 2/6 | 0/6 | 2/6 | .4242 |
| Down128/up256 | 0/6 | 1/6 | 0/6 | 0/6 | .9631 |

The uncompressed authored partial marks were localized closely. JPEG weakened
recovery substantially and produced responses on carriers/ordinary edits with
no added watermark. The full16-cell image/video aggregate includes full-mark
IoU and selected area; no result is presented as generalized sensitivity or
verified customer accuracy.

## Before, during and after frames

Three moving clips contain eight frames at2fps. In the partial state, the
intended mark occupies the center region only on frames2:6, source1–3seconds.
Both raw RGB and H264CRF23 inputs retain exact frame index, PTS and rational
timebase. Before0.5s, during2s and after3.5s frames from all three carriers were
visually checked against the intended region and saved prediction overlay.

| Clip representation | Unmarked-clip frames flagged | Marked frames missed | Partial-clip frames flagged outside intended interval | Mean marked-frame spatial IoU |
|---|---:|---:|---:|---:|
| Raw RGB | 1/24 | 0/12 | 0/12 | .9192 |
| H264CRF23 | 2/24 | 3/12 | 6/12 | .2364 |

Frames are correlated and reuse the same authored patterns. In partial H264
clips, flags outside the intended1–3s interval cannot distinguish detector false
responses from codec/temporal propagation of watermark-like signal. These are
errors relative to the planned source interval, not proof that those later
pixels/frames were independently edited or AI-generated. Exact source timestamps
remain valid even when localization is inaccurate. Missing a public mark cannot
establish originality.

## License, loader and resources

The explicitly MIT SA-1B object was verified against the pinned publisher mirror:
SHA25690ef232384e023bd63245eb0c131abd69d2afc7b8f17a71ccedceb542bf009e2,
377825938bytes. The separately noncommercial COCO checkpoint was excluded.
Publisher references: [source/license](https://github.com/facebookresearch/watermark-anything)
and [model release](https://huggingface.co/facebook/watermark-anything).
Reviewed source/config identities and the adapted-component notices were retained.

The checkpoint was held in RAM, not stored on local disk. Restricted tensor-only
loading required no extra pickle globals; all432 final state tensors matched
the pinned original. The model has94,416,643 parameters. No unrestricted upstream
loader, package installation, serving dependency/model or customer input changed.
Exact neural replay requires reacquiring the pinned checkpoint.

Native execution took37.50seconds with sampled process-tree RSS lower bound
1,801,830,400bytes. Separate replay took1.58seconds at143,343,616bytes. Both
completed without guard activation or sampling errors. Measurements are sampled
lower bounds, not guaranteed peak system/GPU totals. The original freeze-only
attempt stopped on an installed OpenCV distribution-name mismatch before models
or cases ran; the retained record explains the corrected inventory label.

## Separate replay and public reproduction

The model-free review checked all168 identities/input RGB arrays, intended
partial spatial/time boundaries,11,010,048 presence pixels,352,321,536 losslessly
packed binary bit predictions, independent component boxes/majority/overlap
arithmetic and48 re-decoded frames from six MP4s with exact source timestamps.
Every aggregate denominator matched. Continuous bit logits were unused and
not retained; the supported hard message readout is fully replayable from
retained binary maps. **Outside human reviewers remain0.** This is separate
code/arithmetic review, not independent human validation.

Six model-free tests and Ruff passed. The standalone bootstrap is public, so an
outside reviewer can recreate these procedural inputs and assets without access
to private research photos/voice data. It acquires only pinned source files,
licenses and small isolated dependencies; the native checkpoint loads later in
RAM. It refuses existing output folders and verifies every retained source byte.
No source package setup script is executed.

```sh
PYTHONPATH=backend python -m scripts.acquire_public_visual_watermark_v1 /tmp/visual-assets
PYTHONPATH=backend python -m scripts.run_public_visual_watermark_v1 freeze /tmp/visual-assets /tmp/visual-assets/source /tmp/visual-study
PYTHONPATH=backend python -m scripts.public_visual_watermark_runtime_v1 run /tmp/visual-study/profile run /tmp/visual-study
PYTHONPATH=backend python -m scripts.public_visual_watermark_runtime_v1 review /tmp/visual-study/audit-profile /tmp/visual-study /tmp/visual-study/audit
```

Use the recorded framework environment and source/asset identities; codec builds
and platforms can change decoded bytes/results. Run a new study directory rather
than overwrite a retained run. Source bootstrap execution and verification are
recorded separately from the native experiment. Raw authored arrays/private
journals remain local; code, methods, hashes and aggregates are public.

Next: fresh rights-cleared natural-image/real-clip qualification with the point
frozen, plus separately qualified VideoSeal/PixelSeal and text schemes. The newest
PixelSeal checkpoint's inspected size is1,237,429,197bytes; reading its paper/code
does not establish compatibility with private Muse deployment. No serving or
confidence promotion follows from this authored contract. GCP remains last.

The separate [Linux native replay](independent-replay-2026-11-08.md) now also
completed this immutable authored contract and its model-free audit. The
cross-platform comparison retains changed H264 flags/misses and small numerical
differences. This does not replace the Mac results above or complete the
natural-media/provider qualification.
