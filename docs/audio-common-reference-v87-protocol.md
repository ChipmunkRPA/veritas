# Parallel shared-reference calibration comparison v87

Add one low-cost calibration-reference alternative to the completed v84 fits
**before any selection or final-role forward**. The complete fitting/calibration
responses were already available when this alternative was defined. It reuses
calibration data for its intended development purpose, not as a new independent
validation set. Freeze and publish this method, code, tests, parent artifact and
three-way selection rule before its calibration call and before held-role
extraction. Keep all primary v84 methods, data, fits, selection and results intact.

The completed v84 context method routes by its fitting-only flatness cutoff and
balances reference truth/family/group/source/variant weights separately within
each context. That intentionally imposes a 50% AI reference prior within each
context. Its measured context distribution changes from 500/499 assessed fitting
windows to 641/367 calibration windows. This does not prove either calibration
assumption is wrong, but motivates a comparison of reference populations.

The added method **keeps exactly the existing context score coefficients and
cutoff**. It changes only the two calibration maps. Compute the same balanced
weights once over the complete assessed calibration role, then condition and
renormalize those weights within each context. Do not rebalance its classes or
families again within the branch. Its conditional class prior therefore follows
the common weighted reference population instead of being forced to 50% in
every context. Use the same positive slope/intercept bounds, ridge, solver and
minimum three connected calibration groups per truth class. Unsupported contexts
stay unassessed and cannot borrow calibration.

This can retain an association between measured quality and class in the
reference pool. That association may be a nuisance shortcut under a different
background/source distribution. It is an alternative to test, **not a presumed
correction or more reliable probability**. Include a context-prior-only control
outside the selection pool to reveal how much the measured context alone can
do. Publish its categorical counts and AUROC, not learned branch-prior values.
Do not deploy that diagnostic as a content detector.

Predictors remain the two actual component scores and measured input flatness;
truth, provider, known SNR, condition, source ID and role are not inference inputs.
Fitting and calibration source/group separation remain mandatory. No new neural
model, weight download, preprocessing, VAD, source substitution or neural call
is introduced. Full/compact journals and their exact sample clocks remain the
primary v84 records. New maps and all record/voice data remain private.

Use the same 1,044 selection cases and then the same 1,044 final cases, not a
second cohort. Reproduce every original primary method count and add the new
method plus all 36 condition/family diagnostic cells. Report losses, ranking,
reliability, both wrong directions, missed/inconclusive/unassessed and recovered/
lost correct cases with the existing limitations. Class-only bins do not test
probability calibration; shared backgrounds/variants do not support independent
confidence intervals or a population reliability claim.

Preserve primary v84's two-way selection. For this parallel comparison, lock a
three-way pool: global, original branch-balanced context, added common-reference
context. Use exactly the same full-role balanced Brier ranking with neutral
0.25 loss for unavailable scores; missing outputs never acquire a probability
or an original verdict. A method needs one supported calibrated branch. Prefer
global unless the best context improves loss by more than 0.005. Choose the lower
loss between the two equal-complexity contexts, preferring original context on
an exact tie. There is no no-loss/zero-error admission rule or automatic default.

Publish both exact fixed choices and remotely verify both receipts **before
any final-role forward**. The sequential dispatcher checks both receipts before
calling unchanged primary extraction commands. It refuses existing attempts,
stops on the first failed stage and does not retry. All heavy stages retain the
existing corrected observer and unchanged resource limits. After confirmation,
report the preselected choices and all methods transparently; do not retune using
final results. This is one shared held cohort with parallel analyses, not two
independent confirmations. No serving/default, workflow, outreach, real payment,
model-privacy or GCP change is part of this study.
