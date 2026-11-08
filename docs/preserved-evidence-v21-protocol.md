# Preserve evidence identity and check fitting reliability

Prospective consumed-scalar development protocol. No new media, native model
forward, fresh reserve, customer input, serving change or external human review.
The full detection-improvement goal remains open; this is a necessary numerical
and combination comparison, not a replacement for fresh/full-app testing.

## Numerical repair

Leave legacy v2/v3 code and published results byte-preserved. Test a new small
bound-constrained Newton fitter on the exact scalar-head, combination-response
and combination-calibration problems from every published audio-v15, actual-image-
v11, video-v20 and scoped-writing-v6 arm. Reconstruct response/calibration inputs
using their original frozen models, isolating the optimizer change. This includes
all head inputs in the passage experiment that had two actual Linux failures;
the old logs do not uniquely identify the failing context, so test them all.

Keep source/class balance, minimum25 source groups/class, 50:50 reference prior,
source-count target smoothing, response standardization, clipping and L2 .001.
At most8 features. Enumerate active sets for the bound-constrained quadratic
Newton step, at most100 iterations and60 Armijo backtracks. Stop at vector KKT
residual1e-10; independently recompute scalar sigmoid/logloss and gradients with
`math.fsum`, requiring feasibility and KKT infinity residual<=1e-9. This is a
numerical check under floating-point tolerances, not an exact arithmetic proof.
An iteration/line-search/check failure produces no fitted artifact and no retry.

For each original supported problem require the new independent objective no
worse than the old objective plus1e-11. Record original/new gradient residual,
objective difference, source support and logit drift. A legacy library success
flag is not itself the new check. Unsupported contexts remain unsupported.
No old probability/decision is silently replaced when a more accurate optimum
moves a value near a boundary. All original published results remain intact.

Background: SciPy documents [projected-gradient stopping and separate objective
stopping](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html).
Convex optimality and Newton/backtracking principles are described in
[Boyd and Vandenberghe, chapters5 and9](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).
The implementation is original code. Independent authored finite-difference,
SLSQP optimum, active-bound, repeated-source and failed-fit tests precede data.

## Video combination experiment

Reuse all648 public source-bound v20 scalar records and their existing four
source-disjoint roles. These clips and reserved results are already consumed.
No new generalization, identity, AI pixel authorship or lip-sync claim.
Projection onto appearance-only units must preserve every file's planned-scope
truth; refuse rather than silently dropping or relabeling files if it does not.

Five arms, all using the new checked fitter:

1. `appearance-direct`: separately calibrate local units but use maximum raw
   appearance response as the single file feature. A single-view signed maximum
   is not a way to subtract negative evidence from another view.
2. `appearance-positive`: positive calibrated appearance-logit maximum, to
   isolate the legacy feature floor from view identity.
3. `family-max`: all three v20 views, legacy positive maximum within the family.
4. `family-repeat`: repeat that family maximum three times, with slope penalty
   multiplier3 for every copy.
5. `view-max`: positive maximum separately for crop, appearance and feature-change
   views, jointly fitted without any independent-vote/product assumption. Same
   three dimensions and penalty multipliers as `family-repeat`.

Repeating an identical feature with multiplier3 leaves the optimum aggregate
slope penalty equal to the one-feature control. Require maximum probability
difference<=1e-7 between `family-max` and `family-repeat`, with identical available
files. This control separates information preservation from merely weakening
ridge regularization. Negative evidence in another area cannot reduce a view's
positive maximum. Every required unit must be assessed; missing evidence remains
inconclusive. No new missing-view fallback or subset-distribution claim is made.

Fit local heads on specialist_cal, combinations on fusion_fit, and their scalar
calibration on fusion_cal. Freeze every arm/artifact/point before scoring the
reserved role with this new method. Raw reserved measurements are blinded in
the fitting manifests, and later perturbation must leave the fit unchanged.

Select the standard internal point from the same common completely assessed
genuine fusion_cal source groups across all five arms, minimum25. The false-
component budget is the existing raw appearance cue's count at0.11243900750879454,
capped at floor(.1*n). Strict greater than sorted group maximum at rank n-budget.
Report actual held-out false alarms; this is not a public screening threshold.

Also report a predeclared **raw-point-preserving** operating control only for
`appearance-direct`: map the existing raw point through its two positive-slope
file calibration stages. Both mapped standardized values must lie strictly
inside their clip limits; otherwise mark that control unsupported. Require its
flags to equal the raw cue wherever assessed. This is a confidence-calibration
reference, not new detection gain or a second chance to choose a passing point.

Report all160 reserved files:80 manipulated/80 genuine, source-component false
flags, missed AI including inconclusive, coverage and source/class-balanced
Brier. Compare against the stronger existing raw appearance cue (62/80 AI,
1/80 genuine false flag,5/160 inconclusive), not merely the weak crop reference.
Record source positions for every local outcome. No new localization method or
pixel mask is claimed; existing before/during/after evidence still needs its own
full-app check before a later release.

Fresh detection follow-up requires strictly more AI clips than the stronger raw
appearance reference, no more genuine false flags/components or inconclusive
files, and lower Brier than the direct appearance calibrated reference. Keep
every denominator and freeze. A calibrated reference preserving identical raw
flags may support a future confidence-validation study, not claim more edits.
No serving promotion follows automatically, even if a development arm qualifies.

## Review and publication

Run roles sequentially under6GiB sampled process-tree RSS,1536MiB available
memory/disk, and600seconds per scalar job. Capture exact failures and partial
journals. Source/role/raw-value hashes, code, parameters and internal points are
frozen before evaluation. Local and immutable Linux replays must compare every
numeric output with maximum difference<=1e-6 and exact statuses/counts/decisions.
Test changed reserved measurements, missing required observations, duplicate
views, source leakage and JSON artifact round trips. Keep raw media, face/voice
features, alignments, geometry journals and customer uploads private. Only public
scalar inputs, time positions, aggregate numerical/detection outcomes and code
are used or published. Linux numerical agreement is not outside human review.
