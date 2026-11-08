# Image context precision and region calibration: factorial development

The v14 head and strict mean audit remain rejected/failed with their original
source, outputs and records unchanged. This is a new prospectively frozen
development study, not a retuned v14 result or fresh-source confirmation.

## Two changes and one primary candidate

1. Compute full-image feature means using float64 accumulation, then round once
   to float32 for the same small head. Refit that head with the identical seed,
   architecture, fit roles, 30 epochs and optimizer used in v14. The backbone,
   normalization, sampled cells, source truth and image inputs do not change.
   All 840 full fit-feature cases and sampled targets must replay exactly before
   refitting. Store only new context means; reuse the existing private vectors
   and targets read-only, avoiding another 503 MB stored copy.
2. Calibrate the decision that actually produces a displayed region. For a
   median-filtered 32x32 grid, compute the highest level at which an 8-connected
   component of at least four cells exists. Activate equal-valued cells together.
   A selected region exists exactly when this activation is strictly greater
   than the chosen point. Points are actual float32 grid values, preserving
   well-defined equality in the native comparison.

An isolated high cell can influence a raw maximum without surviving the display
component rule. This motivates the new calibration hypothesis; it does not
establish that it caused v14's weak detection or promise an improvement.

Compare all four arms on the same inputs: old head/maximum, old head/regions,
stable-context head/maximum, and stable-context head/regions. The **primary
candidate is stable-context head/regions**, chosen before any new project-image
scoring. Other arms are ablations and cannot be substituted after results.
Old-head arms retain their known original precision failure and cannot become
qualified serving candidates through this study.

## Fixed data, supervision, points and outcome rules

Retain all consumed 288 source groups and 1,344 case conditions from v14:
120 fit groups / 840 variants, 60 point groups / 180 variants and 108 diagnostic
groups / 324 original/edit/ordinary variants. Every source and derivative remains
in its role, and all input/truth/native-feature hashes must match the original
journal. No fresh, unseen-generator or population-accuracy claim is made.

The existing Apache-2.0 register-base backbone/code and 295,041-parameter head
architecture stay fixed. Native input is bilinear-warped 448-square RGB with
ImageNet normalization and normalized blocks 4, 8 and 12. Use CPU float32/two
threads, deterministic algorithms, seed 20261108, AdamW lr .001/weight decay .01,
30 epochs and batches of 512. The 53,760 sampled soft source-area targets and
balanced soft-class loss remain. Never train the backbone or select checkpoints
using point/diagnostic data. Dataset rights and pretrained overlap limitations
from v14 remain; raw photos, pairings, feature vectors and weights stay private.

For each head, fit both maximum and region points from the same 120 genuine
point-role original/ordinary grids. Use the maximum across the two variants
per source, descending rank floor(.05*60)=3, strict greater-than. No AI grid
selects a point. Freeze both head identities, new mean-array hash and all four
points after 960 new cases, before new point-role AI or diagnostic scoring.
Existing data are consumed; reuse does not create a fresh holdout.

Median3 reflect filtering and source projection stay unchanged. Retain
8-connected components of at least four cells. Source pixel (x,y) maps to
floor(32*x/width), floor(32*y/height). Measure both comparison-value exceedance
and actual displayed-region flags. Region-based points must obey their original
three-of-60 calibration-group budget under actual display selection. Exact ties
do not flag. These are internal ranking points, not AI probabilities or public
screening claims. Empty masks cannot establish originality.

Report the same source-bound current-selector reference, all four arms and each
generator/cohort. All 108 edited, 108 genuine and 108 ordinary diagnostics stay
in denominators. Count useful masks with IoU>=.3, mean IoU/pixel F1 including
misses, false highlights and ordinary-control flags. Preserve every subgroup
loss. Primary continuation requires higher mean edited IoU, more IoU>=.3 cases,
no extra genuine flags versus current and ordinary-control flags<=5%, plus the
complete audit below. No successful ablation/subgroup can replace a failed
primary result. Current-selector ordinary-control inference is unassessed until
a passing primary development comparison, then must precede fresh confirmation
and full-app/evidence-export verification. No serving admission follows here.

## Native qualification, complete replay and resources

Before project scoring, repeat four self-authored procedural native inputs with
exact full-forward hooks, independently grouped float64 means within the
unchanged 1e-5 bound and one exact repeat: nine backbone forwards. Publish and
remotely verify code/protocol and a source-panel freeze before new scoring.

Native scoring repeats all 1,344 source cases, 18 full-hook probes and one exact
first-source repeat: 1,363 backbone forwards, plus 3,150 separate head optimizer
updates and 504 new-head response grids. Old-head maps are reused with their
exact source/array hashes. Recompute all 2,016 point/diagnostic mask outcomes
across four arms; old-head/maximum must equal every canonical v14 outcome.

A separate worker repeats all 1,344 native feature cases and 18 full hooks,
all immutable sampled vectors/targets and new means, both heads' 1,008 exact
response grids, source truth and all 2,016 mask/metric sets. Independently find
activation levels using binary search with flood fill, rather than the primary
union algorithm, and independently refit all four genuine-only points. All
old v14 means must reproduce, retaining their 1,302 known original failures.
New means must satisfy the original 1e-5 reference bound; independent dense-head
float64 arithmetic retains 3e-5. The optimizer is not independently refitted and
no outside human reviewer is claimed. Preserve every incomplete/failed phase.

Run each local heavy phase alone with a 900-second guard, at most 6 GiB sampled
process-tree RSS, and at least 1,536 MiB available RAM/free disk. Sampled RSS is
a lower bound; retain sampling errors and guard status. No implicit retry,
case omission or model/point amendment after scoring. Publish methods, source
commitments and aggregate receipts only. No customer inputs, provider verifier
requests, billing/default-confidence changes or GCP resources are involved.
