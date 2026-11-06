# Direct mouth-artifact head: development result

Decision: **reject**. This new fitted head flagged two AI clips, but neither
flag correctly localized the known mouth edit. It also flagged three non-AI
controls. It failed the frozen localization/spatial-overlap requirements and
is not enabled. No fresh confirmation or outside review was performed.

## Source-separated training

The [protocol](mouth-trained-v6-development.md) fixed the representation,
regularization, source roles, review-point recipe and development gates before
feature extraction. It used 20 previously consumed CREMA-D actors: 16 for
fitting, four for selecting a raw-margin review point. The eight v4/v5 actors
were excluded from both roles and retained only as consumed development.
These are actor-separated roles, not independently recruited reviewers or a
new untouched confirmation dataset. Four source sentence types overlap.

All **800 planned native training frames** were prepared: 640 fit and 160 point
frames. Each selected source frame has an original, local brightening,
non-neural resampling and two known neural mouth variants. Conditions share
the same source positions and mouth support. No actors/frames were substituted.
PNG and exact changed-pixel masks were retained locally; originals were untouched.

The representation is a native mouth/lower-face chip, followed by frozen
DINOv2-small CLS/mean-patch features and a 768-input regularized linear head.
Fitting balances actors and classes. The encoder's pinned
[checkpoint card](https://huggingface.co/facebook/dinov2-small/blob/ed25f3a31f01632728cabb09d1542f84ab7b0056/README.md)
and [official source licence](https://github.com/facebookresearch/dinov2/blob/main/LICENSE)
declare Apache 2.0. Existing permitted YuNet and the v4 generator/geometry are
reused. CREMA-D retains its separate ODbL/DbCL terms. Publisher declarations
do not establish full training-data rights or pretraining independence.

The fit converged in 110 iterations. Only the four point actors' 96 non-AI
frames selected the point, 1.5281023444548434 **raw-margin units**. Their AI
responses and the eight development actors could not tune it. This is not
probability calibration, a public screening cutoff or customer confidence.

## Complete video development results

| Condition | Planned | Baseline / candidate correctly localized AI | Candidate AI clips flagged | Baseline / candidate non-AI flags |
|---|---:|---:|---:|---:|
| Same-frame AI mouth redraw | 8 | 0 / 0 | 1 | — |
| Earlier-target AI mouth redraw | 8 | 0 / 0 | 1 | — |
| Untouched control | 8 | — | — | 0 / 1 |
| Local brightening | 8 | — | — | 0 / 0 |
| Non-neural crop/resizing | 8 | — | — | 0 / 1 |
| Constant audio delay | 8 | — | — | 0 / 1 |

All 48 executions were assessed without failures, and all baseline windows and
source centres exactly replayed the frozen v4 receipts. Fourteen AI clips had
no flag; **all sixteen AI edits lacked a correctly localized range**. Mean
temporal IoU over all AI cases was 0.0125 and spatial IoU was zero. The point
remained fixed from separate actors, so these development-control flags are
not zeroed by construction. Counts are correlated variants of eight actors,
not 48 independent subjects.

Every flagged case belonged to actor 1008. The selected range was **0.48–1.00 s**
in the original, resized, delayed and two neural versions. The actual neural
edits occupy **0.88–1.68 s**; temporal IoU is only 0.10. The representative
chip is frame 17 (0.68 s), before the edit starts at frame 22. Thus an AI-file
flag cannot be counted as successful evidence localization. The identical
untouched/delayed flags further show sensitivity to that source's ordinary
appearance. The experiment does not establish a general causal explanation
for the head's failure.

The allowed one-extra-control-flag-per-condition gate passed; the required
three additional correct locations and 0.01 mean spatial improvement failed.
[All outcomes](mouth-trained-v6-evaluation.json) remain reported. The current
app's voice, other visual/provenance checks and confidence calculation are
unchanged; this is not a whole-app accuracy measurement.

## Source audit and public numerical replay

Separate local audits passed all **800 training inputs** and **2,016 native
video inputs across 48 clips**. They re-decoded original sources, checked
training masks/labels/positions, and recomputed face geometry, mouth chips,
normalized float32 input hashes and retained training-vector hashes. They
verify correspondence, not independent human review or detection quality:
[training audit](evaluation-panels/mouth-v6-training-audit.json),
[video audit](evaluation-panels/mouth-v6-video-audit.json).

The [fitting capsule](evaluation-panels/mouth-v6-fitting-capsule.json.gz)
contains allowlisted source identities, 800 frozen numerical feature vectors,
and the fitted head/normalization. It includes the separate data attribution
and terms; no source media, private paths or pretrained vision weights are
included. An outside reviewer can refit with NumPy/OpenCV/scikit-learn and
inspect source-role separation without downloading the vision model.
The local [CPU replay](evaluation-panels/mouth-v6-fitting-replay.json) exactly
reproduced the coefficients, normalization, intercept and review point: all
recorded maximum differences were zero. This does not authenticate the neural
feature extraction or constitute outside review. See the
[replay instructions](evaluation-panels/README.md).

All 48 focused research tests passed. Lint and whitespace checks passed.
No Linux/CUDA/GCP verification, customer evidence integration or paid flow is
claimed for this rejected head. No runtime configuration was changed.

- Generation plan: `b6b7a98bbc148668a3ffcd4377c08568634f849a54cd0b61979c2f99381f8933`.
- Training manifest: `c0126feb234f4a4cd673204f8fa3c21b7a9e95fb5f57290dbd4f6af2f4379522`.
- [Comparison lock](evaluation-panels/mouth-v6-lock.json): `78acad13eb49139293e8cf82b76b44dfb9a39267a546f0c19493ef8aba04118d`.
- Fitted head: `b8c608cfc8b1c0fd990665d9a8956d07b6aa8ea3c9dd60bc8a87e481a1404b06`.
- Video score receipt: `bee22a3783132d3a64d9c5fdbd10e8e107560fdf86974ef3570d1e5b03f2b211`.
- Public capsule: `11f6c6dcdeb755aa25aee8b01498f3c59619176abd886bd50fc533584831cfb4`.

Further work needs more diverse, permitted training/editing families and a
representation capable of learning manipulation traces without ordinary-source
shortcuts. Do not lower this failed head's point or score fresh actors to rescue
it. New confirmation must keep actors and sentence families unseen and preserve
realistic low-resolution and ordinary-edit controls.
