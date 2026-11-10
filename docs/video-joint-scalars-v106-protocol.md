# Retain check identity in the video face combiner

The earlier [v20 development result](video-face-specialists-v20-result.md)
found that its calibrated family-maximum combinations were weaker than the
existing raw appearance reference. This follow-up tests the stated pooling
limitation on the **already consumed public scalar release**. Prior results
are known. It is post-result development, not fresh confirmation or a replay
of native inference. The new 90-group source panel is not used for fitting or
selection here, and its ongoing construction is unchanged.

## Fixed comparisons

Retain the exact released local heads for crop, appearance and patch-feature
changes. They were fitted on the original `specialist_cal` role and keep their
separate local targets, constants and parameters. For each selected view, take
its largest calibrated logit across all planned units and clamp below at zero,
as in the old feature readout. Preserve a separate feature per view rather than
one maximum across the dependency family. The checks remain dependent; their
probabilities are not multiplied as independent votes.

Fit two file-scope arms:

1. Appearance-only, one retained feature, providing the missing stronger
   calibrated reference.
2. Joint crop/appearance/feature-change, three retained features.

Use the unchanged source/class-balanced monotone logistic fitter, nonnegative
slopes, target smoothing, L2 .001 and minimum 25 known source groups per class.
Fit each response on `fusion_fit` only. Fit its separate one-dimensional
calibration on `fusion_cal` only. Keep a 50:50 reference prior; this is not a
customer-specific probability of correctness. Preserve every unavailable
required unit: its arm remains unassessed rather than imputing zero. The
appearance-only arm may remain assessable when another check is unavailable;
the joint arm may not silently substitute it.

## Research points and evaluation

Use the intersection of genuine `fusion_cal` components for which every genuine
file is completely assessed by both new arms and the historical all-check
combiner. Require at least 25 such components. Fix a budget of three components,
the original v20 budget. For each new arm, use the sorted common group maximum
at rank `n - 3 - 1`; flag strictly above it. Ties may produce fewer flags. Do not
tune this point on the evaluation role, invent a public threshold claim, or
change it after results. Keep the old combiner's published point and the
existing raw appearance point .11243900750879454 unchanged.

Seal fitted models/points before calculating the 160 `confirmation`-role file
responses. This role is already consumed development data despite its historical
name. Report every file, AI misses including inconclusive files, genuine false
flags and source components, coverage and source/class-balanced Brier on scored
cases. Binary raw flags have no invented probability calibration or Brier.
Reproduce the old all-check and raw-appearance reference counts from the exact
release before interpreting the new results.

This is file-scope presence within the original planned face supports. Maxima
for different views need not occur at the same instant. No new timestamp or
localization probability follows from the file score; current local measurements
retain their own support. Temporal localization and the full application require
the separate native common-input study.

Show useful gains together with both error directions and missing coverage.
A moderate additional false-alarm count is not an automatic rejection. No arm
is automatically admitted to serving: new source/native inference, calibration
and complete-app evidence/export/payment qualification remain necessary.

## Execution and publication

Freeze code, tests, procedure, fitter, source-release/artifact/point hashes and
NumPy/SciPy/psutil versions. Publish and verify the freeze before fitting. The
exclusive single-process numerical stage uses 900 seconds, 1.5 GiB self RSS,
3 GiB initial available RAM/disk and 1.5 GiB running reserves, observed between
phases rather than continuously. Pin numerical thread settings to one. No neural
runtime or model is loaded, and original neural budgets are unchanged.

Keep new learned coefficients and per-file response journals private. Publish
the complete aggregate comparison, code and whole-artifact commitments. Preserve
all original studies and any failed attempt; no automatic retry, overwrite or
response-dependent source replacement. Public v20 scalars permit later outside
reproduction without clips or new inference. No outreach, Actions, charge or
GCP action is part of this study.
