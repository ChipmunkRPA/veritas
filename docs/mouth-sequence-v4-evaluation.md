# AI mouth editing: source-bound development comparison

Decision: **reject the mouth-region statistic**. The running application,
detector defaults and confidence calculations remain unchanged. No fresh
confirmation actors were scored and no outside review has taken place.

The [generation protocol](mouth-edit-v4-development.md) created 48 cases from
eight previously consumed CREMA-D actors: 16 known AI mouth edits and 32 non-AI
controls. This is development, not an independent accuracy test. The
[comparison protocol](mouth-sequence-v4-comparison.md) froze code, regions and
gates before scoring. Both methods used identical production face tracks,
source centres, native triplets, aligned crops and patch tokens. The original
mask calculation exactly matched all 48 production-adapter captures. There
were no scoring failures or unassessed sequence cases.

## Known edits and controls

Edits use the MIT-declared human LivePortrait weights with permitted mesh
preprocessing; the restricted InsightFace detector is excluded. Only the mouth
support is composited from neural output. Before/after frames and outside
pixels remain unchanged. Same-frame and eight-frame-earlier lip ratios are
requested conditioning, not proof of perfect synchronization or exact achieved
lip delay. Both are known AI edits irrespective of measurable timing changes.

Controls include untouched video, local non-AI brightening, non-neural
crop/resizing and a constant 320 ms audio delay. The visual controls use the
same support and interval as AI edits. All cases share a lossless video/PCM
recipe. A non-AI label does not imply an unedited file.

The first preparation found partial final audio frames in four originals:
24 cases were available and 24 unavailable. Generation 4.1 used complete shared
audio/video spans, preserving all eight selected sources and the selection seed.
Four unchanged complete preparations were reused after file, mask, generator,
frame and sample-count checks. The initial plan, code, protocol and failure
receipts remain separately retained. Detector outcomes did not guide this fix.

## Complete outcomes

| Condition | Planned | Current statistic: localized AI | Mouth statistic: localized AI | Current / mouth non-AI flags |
|---|---:|---:|---:|---:|
| AI redraw, same-frame target | 8 | 0 | 0 | — |
| AI redraw, earlier target | 8 | 0 | 1 | — |
| Untouched control | 8 | — | — | 0 / 0 |
| Local brightening | 8 | — | — | 0 / 0 |
| Non-neural crop/resizing | 8 | — | — | 0 / 0 |
| Constant audio delay | 8 | — | — | 0 / 0 |

The current statistic missed all 16 known AI edits under this localization
rule; the candidate missed 15. One additional localization failed the frozen
three-case requirement. Mean temporal IoU over all AI cases rose from 0 to
0.056818, and spatial pointing IoU from 0 to 0.007178, below the required 0.01
gain. Coverage and retention gates passed. The new point was
0.008061921782791618 in raw feature-distance units, chosen just above the largest
non-AI development response. Zero control flags are by construction on those
same controls, not a validated future false-alarm rate. This is not an AI
confidence probability or public screening threshold.

The existing audio/mouth timing adapter separately flagged none of the AI edits
or controls. Each eight-case condition had seven `ok` and one `not_applicable`
status; `ok` does not guarantee a stable offset or general lip-forgery detection.
Timing windows and metrics remain in the local receipt. These outcomes concern
the specified sequence/timing checks, not the application's other visual,
voice, provenance or combined assessments.

## Source audit and reproducibility

A separate local audit decoded all 48 final cases and checked every frame,
timestamp, changed-pixel mask, unchanged surrounding frame and soundtrack.
All 48 passed. This verifies known benchmark transformations, not unknown
customer media or outside human review.

The 16 focused regression checks passed, covering immutable asset downloads,
preservation of another attempt's partial file, exact composite support,
lossless frame/audio round trips, production statistic parity and complete
evaluation denominators. Lint and whitespace checks also passed. These are
research-tool checks; no new detector was enabled in the application.

- Generated manifest SHA-256:
  `d652d3d67ff9bc670ed0e7b18eecc0958daf71fcd6a05c55f0dfd2c16b32a0b2`.
- Comparison lock SHA-256:
  `3fd086f99876ed74ed7891e8ab48c625cd3426f8ae5010ddaa09b81fdb6d0660`.
- Score receipt SHA-256:
  `da28a3e2aa26b9422d5f9dee630a6a55553ae6611bc8d063347db2e8011c4551`.
- [Aggregate/case results](mouth-sequence-v4-evaluation.json),
  [source/input catalogue](evaluation-panels/mouth-v4-development.json),
  [source audit](evaluation-panels/mouth-v4-source-audit.json), and
  [renderer pins](evaluation-panels/mouth-v4-renderer-assets.json).

The [source licence](https://github.com/KlingAIResearch/LivePortrait/blob/9b294b3d0536135442ea73cb01e6cb3ca7029dd3/LICENSE)
requires replacing InsightFace detection models for commercial use; the
[checkpoint card](https://huggingface.co/KlingTeam/LivePortrait/blob/82a4fa6735ca58432b6ce39301b4b9ee066dea47/README.md)
declares MIT. The official model cards for
[BlazeFace](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf),
[Face Mesh V2](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf), and
[blendshapes](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf)
declare Apache 2.0. CREMA-D retains its separate
[ODbL/DbCL declaration](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt).
These are recorded publisher permissions, not a complete training-rights audit
or proof of pretraining independence. No restricted face model, media, mask
image or checkpoint binary is published in Git.

The renderer passed strict safe-state loading and a three-frame CPU smoke.
With MPS explicitly selected, warm renders took about 0.7 seconds each on the
local Apple M3 Pro. An earlier attempted GPU smoke omitted that option and
actually ran on CPU; its receipt records CPU and remains separate. CPU/GPU
pixels differ, so regeneration must record new hashes and recompute masks.
The driver is tested on this Mac, not verified as a GCP runtime.

## Next methodology

Timing agreement and this regional-change statistic are insufficient as reliable
AI-lip-edit detectors. Existing customer interpretation already treats timing
as context and explains that redrawn mouths can match audio. A stronger learned
visual-artifact method and a separately evaluated audiovisual-correspondence
method remain future work. Each needs a new frozen design, ordinary controls,
fresh actors/sentences and exact source-position evidence. This rejected screen
must not be reused as fresh confirmation or enabled because one case was found.
Only eight actors, controlled studio footage, shared sentence templates and one
generator family limit generalization.
