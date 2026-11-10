# Separate appearance calibration beats the old pooled combination

**Advance the appearance-only calibrated reference for fresh native testing,
and retain the joint arm as a comparator.** Both find 64/80 manipulated clips
on this consumed development panel, but appearance-only has fewer genuine false
flags and inconclusive files. The existing raw appearance cue remains a strong
reference: it finds 62/80 with only one genuine false flag. No serving default,
pretrained weights, production local review point or customer confidence changes here.

The [fixed procedure](video-joint-scalars-v106-protocol.md) is explicitly
post-result development on the public v20 scalars. Prior results were known.
Its code, tests, fitter, source package and numerical versions were published
as `483c57a`; remote freeze blob `b3c76353ec4cd6c307b0fddfd36a86415355a8c8`
was verified before fitting. No new native/model inference was performed.

## File-scope results

Each row evaluates the same 160 files from 40 known source components: 80
manipulated and 80 genuine controls. Missing required observations stay
inconclusive and count as missed AI where applicable. File false flags and
source components with any false flag are different denominators.

| Method | AI flagged /80 | Missed AI, including inconclusive /80 | Genuine false flags /80 | False-flag components /40 | Inconclusive /160 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing raw appearance cue | 62 | 18 | 1 | 1 | 5 |
| New appearance-only calibrated file score | 64 | 16 | 3 | 3 | 5 |
| New joint per-view calibrated file score | 64 | 16 | 4 | 4 | 9 |
| Historical family-maximum calibrated combination | 29 | 51 | 5 | 4 | 9 |

The historical pooled-combination and raw-appearance counts reproduce exactly.
The appearance-only arm adds two AI flags and two genuine flags relative to raw
appearance, losing none of its AI flags. The joint arm also adds two AI flags,
losing none, but adds three genuine flags. These are secondary paired diagnostics
from the [arithmetic check](evaluation-panels/video-joint-scalar-arithmetic-v106.json).

The new research points permit at most three flagged components among the same
39 completely assessed genuine calibration components. Both reach that budget.
The old raw cue keeps its original fixed point. Thus the two additional AI flags
are an operating-point/calibration tradeoff, not evidence that a new neural
detector has intrinsically improved. A monotone appearance map does not invent
new forensic features. Moderate extra alerts are not an automatic rejection;
they must be evaluated alongside useful detections and coverage on new sources.

## Calibration and score-pooling interpretation

Existing local maps remain fixed. Each view supplies its own maximum positive
calibrated logit. The new joint combiner retains three coordinates instead of
discarding the contributing check's identity through a single family maximum.
Both new file response heads fit only `fusion_fit`; their separate score
calibrations fit only `fusion_cal`. New parameters and points were sealed before
calculating these development evaluation responses. Each class meets the
unchanged minimum 25 fitting/calibration source groups, with source/class
balancing, L2 .001 and a declared 50:50 reference prior.

The source/class-balanced Brier values on each arm's own scored cohort are
.101693 appearance-only, .095568 joint and .201978 historical combination.
Those first two cohorts differ. On the **same 151 scored files**, a secondary
calculation gives **.094559 appearance-only versus .095568 joint**. The joint
arm's smaller own-cohort value therefore does not establish better calibration
than appearance-only. These reused cases and small differences do not establish
population calibration or a customer's probability of correctness.

This comparison concerns presence within the original planned face supports.
The strongest measurements for different views may occur at different times.
A file score does not supply a calibrated suspicious timestamp, whole-video
authenticity, lip-sync result or pixel-origin claim. Temporal localization and
complete-app evidence behavior still need the separate native study.

## Verification and public reproduction

All 49 relevant tests passed, including 11 new combination cases and seven
reproduction-comparison cases. Tests retain missing evidence, require complete
genuine components, preserve exact counts, and show that changing evaluation
responses cannot affect response fitting or calibration. The original release
commitments, local maps, raw appearance point and audio rollback file remain
unchanged.

The numerical stage exited zero in .71 seconds, with maximum observed self RSS
228,704,256 bytes. Phase observations are not continuous or guaranteed peaks.
The 900-second/1.5-GiB numerical bounds and RAM/disk reserves were unchanged.
It loaded no neural runtime or pretrained weights.

A separate `math.fsum` readout of the saved parameters reproduced all 640
file/arm responses, including 28 unassessed observations, with maximum absolute
probability difference 1.67e-16. This is separate probability arithmetic using
the retained fits, not independently written fitting software or native inference.

The [fresh-directory reproduction](evaluation-panels/video-joint-scalar-reproduction-v106.json)
then refitted both models and their calibrations from the public data. Fitted
artifact and prediction-journal hashes matched **exactly** on this pinned local
runtime. Aggregate counts, research points and fitting support also matched.
This uses the same fitter/runtime; it is not an outside human evaluation.

In the pinned environment, use a destination directory that does not already
exist:

```sh
PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  backend/.venv/bin/python -B -m scripts.reproduce_video_joint_v106 \
  backend/data/qa/video-joint-v106-new-reproduction
```

The driver copies public inputs and the published freeze into the new directory,
keeps fitting outputs there, and preserves the repository's expected results.
It needs no clips, facial vectors or pretrained model downloads. Counts/statuses
must match exactly; floating results use the stated 1e-9 absolute/1e-12 relative
comparison allowance. Different numerical package versions are not silently
accepted as equivalent qualification.

The [complete aggregate result](evaluation-panels/video-joint-scalar-result-v106.json)
contains all arms and whole-artifact commitments. New coefficients, per-file
predictions and private reproduction outputs remain local. The next step is
common native-source extraction and fresh comparison against raw appearance,
with separate calibration, genuine edits and explicit missing-view controls.
The full 90-group input construction is still running. No outside outreach,
GitHub Actions, real charge or GCP deployment occurred.
