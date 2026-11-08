# Image noise/texture features: both candidates rejected

The independently implemented feature localizers do not improve the current
image-region check. They reduce some genuine false flags but miss substantially
more edits and overlap the true edited pixels less often. Neither arm meets the
fixed continuation rules. Do not promote, tune their comparison points using
these diagnostics, or describe this as improved detection.

## Completed development comparison

The [protocol](image-residual-v13-protocol.md) and
[freeze](evaluation-panels/image-residual-v13-freeze.json) were published in
`f8d9adb` before extracting project-image features. All 288 previously consumed
source groups remain source-disjoint between fitting (120), comparison-point
selection (60), and diagnostics (108). This is held out from fitting but is
**not fresh project evaluation data** or an unseen-generator test. It covers
Firefly, SD2 and FLUX Fill in quality-75 JPEGs, with unknown external model/data
overlap and limited ordinary-edit diversity.

The run completed all 1,344 feature cases: 840 fit variants, 180 comparison-point
variants, and 324 diagnostic originals/edits/ordinary controls. It fit 107,520
soft-labeled cells; the projected arm estimated its nuisance directions from
30,720 genuine fit intervention-delta cells. Lighting, recompression and a
conventional blur/tone patch are separate from the exact source edit truth.
The two arms use the same augmented samples and regularization. No pretrained
checkpoint, new model download, neural forward or external upload was used.

## Diagnostic outcomes

Each method has 108 edited and 108 genuine originals. Candidate ordinary-edit
controls are 108 additional genuine inputs. A file flagged here means it has a
selected region, not that calibrated AI authorship was established. A selected
region can miss the true edit entirely.

| Method | Edited files with any region | Edited IoU >= .3 | Mean edited IoU, including misses | Genuine originals flagged | Ordinary controls flagged |
|---|---:|---:|---:|---:|---:|
| Current input-bound selector | 19/108 | 6/108 | .043006 | 7/108 | Not assessed in this study |
| Noise/texture features | 6/108 | 1/108 | .008622 | 4/108 | 2/108 |
| Same features, two nuisance directions removed | 6/108 | 1/108 | .005327 | 6/108 | 4/108 |

Both candidates fail higher mean IoU and more IoU>=.3 cases. Both pass the fixed
genuine false-flag and aggregate ordinary-control rules, but that does not offset
their localization loss. Mean pixel F1 is .064904 current, .013093 feature-only
and .008514 projected. Cases with no selected pixels count as misses.

The 60-source v11 diagnostic subset has 3 current IoU>=.3 cases versus 1 for each
candidate. On the 48 consumed v12 sources it has 3 current cases versus 0 for
either candidate. The conclusions do not reverse between those source cohorts.

| Generator, 36 edited + 36 genuine + 36 ordinary controls | Feature-only: edited flagged / genuine / ordinary | Projected: edited flagged / genuine / ordinary |
|---|---:|---:|
| Firefly | 2 / 2 / 2 | 2 / 3 / 3 |
| FLUX Fill | 0 / 0 / 0 | 1 / 1 / 0 |
| SD2 | 4 / 2 / 0 | 3 / 2 / 1 |

Each candidate's single IoU>=.3 case is SD2. The projected Firefly ordinary
control rate is 3/36, above 5% despite the aggregate control rule passing. No
generator subgroup is selected as a successful deployment path. Full unaltered
cohort outcomes and fixed points are in the
[aggregate result](evaluation-panels/image-residual-v13-result.json).

## Separate replay and limits

The [separate replay](evaluation-panels/image-residual-v13-source-audit.json)
checks all 1,344 planned case identities/roles, reconstructs two genuine-only
comparison points, and independently calculates all 1,008 stored response grids
and source masks across comparison-point and diagnostic cases. It combines
affine coefficients in a different order, uses an independent median-window
calculation, flood-fill components and integer cell-boundary projection. The
largest response difference is 6.66e-16; every source mask and pixel count agrees.
This is separate arithmetic implementation, not outside human review or an
independent provider/dataset accuracy study. It does not independently recreate
every source feature or rerun fitting.

The [execution receipt](evaluation-panels/image-residual-v13-execution.json)
records both successful guards. Feature extraction/fitting/diagnostics took
172.61 seconds with 601,227,264 bytes sampled process-tree RSS lower bound.
Separate replay took 12.37 seconds with 434,012,160 bytes. Neither guard fired;
neither reported a sampling error. These are sampled process measurements, not
total system/GPU memory guarantees.
Sixty-six selected feature, replay and existing image-evaluation regression
checks pass; the new source files also pass lint and whitespace checks.

The result rejects this simple feature/projection proposal. It does not test or
disprove EISL's learned CLIP/LoRA architecture. The next localization method needs
a more useful learned representation and appropriate source/ordinary-edit
supervision before spending fresh confirmation data. The current selector also
remains weak; retaining it is not a claim of reliable pixel attribution.

No serving code, confidence calibration, payment requirement or customer result
was changed. No full-app candidate promotion test or current-selector ordinary
control inference was attempted because the prerequisite development rules
failed. Photos, source pairings, feature packets, scores and fitted heads remain
private. Only code, methods, aggregate results and commitments are published.
Outside human reviewers remain zero, and GCP remains deferred.
