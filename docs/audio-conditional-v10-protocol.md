# Conditional audio response experiment — prospective protocol

Prepared 6 November 2026, after [v9 failed development](audio-identity-v9-development-evaluation.md).
Its optimistic fixed-probability bound rules out repairing five generator families
by guard removal alone. This experiment changes the learned response mapping.
Publish this protocol, implementation and exact measurement freeze before new
project-data fitting or neural forwards. Engineering tests use synthetic fixtures.
No part of the app, overall confidence, payment/auth or GCP setup changes here.

## Unchanged sources and representations

Retain the corrected v9 canonical source roles: 214 fitting originals/1,926
variants, 97 calibration/873 variants and 161 consumed development/1,449 variants.
Keep all known parent-voice, script, byte and PCM connections grouped. Reuse the
exact frozen backbone representations and v9 fitting-only PCA16 projection.
Do not refit the projection on calibration/development or use outcome labels
to change source selection. The fixed 1,080 confirmation inputs remain reserved
and are refused by this runner. Known historical voice exposure and unknown
pretrained overlap remain disclosed in the v9 source record.

Committed original feature JSONL SHA-256:
`315c07c1eec95553760a671e0480fb99bb53669e6f5679be0a73a1077cac512d`.
Projection-head SHA-256:
`411c4322ac35037073f2998ef0757d80c5bd348413d509b302bca90bcdcc7519`.
Complete current-service/probe receipt SHA-256:
`7dff2eb4259f97731bfde0b6e1f3e38a0fb60d9920431212eeb8f9790e8fac9e`.
The served calibration and verified commercially permitted neural weights stay
identical. No new pretrained checkpoint or model download is introduced.

## Separate candidate arms and nuisance control

1. **Nonlinear:** original clipped raw logit and sixteen existing head-orthogonal
   residual coordinates. This tests the compact nonlinear learner.
2. **Quality:** the same seventeen predictors plus two unchanged v5 waveform
   descriptors: bounded spectral flatness and frame-energy floor coordinates.
   This separately tests their additional value. Measure them from the exact
   production PCM without modifying classifier input.
3. **Quality-only nuisance:** only those two descriptors, on the same assessed
   cases. Report this to expose possible recording/source shortcuts. This arm
   is never eligible for confirmation or adoption, even if its metrics look good.

No source ID, generator, dataset, truth label, mixture condition, known SNR or
background identity is a serving predictor. Those fields are used only for
fitting weights, source separation and evaluation strata. A descriptor or useful
correlation is not physical evidence of AI origin.

All three use the same fixed numerical histogram gradient boosting: binary log
loss, 200 iterations, learning rate0.05, depth2, at most4 leaves, minimum40 samples
per leaf, L2=1, 255 bins, all features, random seed461110, no warm start, no
categorical features, no early stopping or validation holdout. Raw-logit response
is constrained nondecreasing for both candidates. Other predictors are unconstrained.
Use original integer labels and unchanged source/class/family/group balancing,
rescaled to sum to the assessed fitting count for tree regularization. Only fitting
roles learn trees; all parameters above are fixed rather than chosen on development.

Use installed scikit-learn1.9.0, without upgrading. Its pinned BSD-3-Clause license
SHA-256 is `fc8dc597df0e3f51da65e2a71021e7f92e0b481998ec572a53c03ac0441db040`;
the freeze commits the local Python/native learner files. The [official license](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING)
permits redistribution and use with its conditions. The [official API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
documents monotonic constraints; actual installed signatures/version are checked.
Keep checkpoint/data terms separately; this software license does not replace them.

Export numerical trees as JSON, without pickle or executable model objects.
Require JSON/native prediction agreement within1e-12 on every assessed fitting,
calibration and development input, using held inputs only for implementation
parity. Save the fitting artifact before independent guard calibration.

## Confidence, probes and comparison

The tree's response is its own source/class-balanced reference estimate.
For each arm separately, calibrate the unchanged worst nonconformity per canonical
group and truth class on calibration roles only. Preserve unassessed variants,
minimum nine groups/class, alpha0.1, internal0.8/0.2 diagnostic points and both
six-decimal raw/mapped original-stability changes within ln(2). These internal
points are not public SaaS confidence or accuracy promises.

Recompute waveform descriptors on each actual filtered stability input. Never
reuse the unfiltered descriptor as the probe descriptor. Exact v9 probe features
may be reused only after source/PCM/head/digest/runtime binding; the quality
preparation verifies their filtered PCM hash. Run the original model for newly
needed probes, retaining every error and unassessed case. No missing probe can
establish an original conclusion.

Compare every arm separately against the exact current guarded service on all
1,449 development inputs. Preserve all sixteen v8/v9 gates, complete denominators,
paired group bootstrap and per-condition/class/generator outcomes. Do not count
more abstention as a detection gain. Report both candidates and the nuisance arm,
including failures; do not hide an unsuccessful arm.

Only candidates passing all sixteen gates are eligible for separate confirmation.
If just one passes, select it. If both pass, prefer the simpler nonlinear arm
unless quality adds at least five correct conclusions with no increase in either
false-call count and no higher Brier error; then select quality. If neither
passes, reject both. This selection rule is fixed before fitting; it does not
enable a model or consume confirmation automatically.

## Completion requirements

Before fresh scoring, require complete source/measurement checks, original/probe
neural-input bindings, JSON/native parity, numerical reconstruction, CPU refit
and the predetermined v9 99-cell native replay. A new passing method needs a
separate exact-input/candidate freeze for confirmation. Paid analysis, evidence,
PDF/JSON and rollback verification still precede any local adoption. Independent
human testing is outstanding and cannot be claimed from local scripts or public
contribution instructions. GCP stays deferred.
