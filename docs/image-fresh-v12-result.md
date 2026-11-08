# Fresh JPEG confirmation: regional candidate rejected

The frozen primary-plus-regions candidate does not meet the replacement rules.
On 48 new source pairs it flags two more edited files than the current component
selector, but also three more genuine files. It flags one fewer edited file than
the primary-only comparator and has higher balanced Brier error. Do not promote,
retune on this holdout or present the development improvement as confirmed.

## Scope and fixed test

This tests editing-presence scores and the actual JPEG localizer's component
selection, **not the whole application's verdict or general AI-image accuracy**.
The current reference replays its production postprocessing using actual native
model responses. Neither calibrated candidate changes which pixels are selected.

All 48 original/edit pairs were selected before inference, excluding 317 known
earlier selected source IDs. Each of Firefly, SD2 and FLUX Fill contributes 16
pairs, split equally between bbox and segmentation variations. Inputs are 96
deterministic quality-75 JPEGs at original source dimensions. These source IDs
are fresh to the known project evaluations; generators are not unseen, and
model-pretraining overlap is unknown. The publisher's original RGB pair
difference supplies local editing truth; JPEG compression adds no AI labels.

The [complete protocol](image-fresh-v12-protocol.md), [freeze](evaluation-panels/image-fresh-v12-freeze.json)
and independent audit code were published in `4bd93fc` before the first new
holdout forward. Same cached commercially permitted assets, original calibration
artifacts and original internal comparison points; no fitting or point selection
used the fresh cases. The [original development study](image-supported-v11-result.md)
remains unchanged.

## Complete file-level outcomes

Each method assesses all 96 inputs. An assessed input means a measurement was
available; it does not establish a reliable whole-file authorship conclusion.

| Fixed method | Edited files flagged / 48 | Edited files missed / 48 | Genuine files flagged / 48 | Unassessed / 96 | Balanced file Brier |
|---|---:|---:|---:|---:|---:|
| Current component selector | 10 | 38 | 2 | 0 | Not a probability score |
| Primary-only calibration | 13 | 35 | 5 | 0 | 0.237222 |
| Primary plus regional calibration | 12 | 36 | 5 | 0 | 0.239117 |

Primary-only has 78 exact comparison-point ties, retained as not flagged under
the frozen strict-greater-than rule. This reflects the positive-feature floor;
it is not grounds to change tie handling after observing results. The regional
candidate has no exact ties. No missing cases or failures were dropped.

| Generator, 16 edited + 16 genuine each | Current: edited / genuine flagged | Primary-only: edited / genuine flagged | Candidate: edited / genuine flagged |
|---|---:|---:|---:|
| Firefly | 2 / 0 | 2 / 0 | 4 / 2 |
| FLUX Fill | 3 / 1 | 5 / 3 | 3 / 2 |
| SD2 | 5 / 1 | 6 / 2 | 5 / 1 |

All three failed rules remain failed: no extra genuine flags versus current,
no lost edited-file flags versus primary-only, and lower Brier versus
primary-only. The other five rules pass. The candidate is rejected as a whole;
no generator subset or alternate point is selected.

The fixed 5,000 source-pair bootstrap gives candidate-minus-current edited-file
flag difference +4.17 percentage points, with a percentile 95% interval of
0 to +10.42 points; genuine false-flag difference is +6.25 points, interval
0 to +14.58. Versus primary-only, Brier difference is +0.001895, interval
-0.005819 to +0.010428. Small paired data and intervals containing zero do not
establish a general improvement. Full rates, coverage differences, local-box
scores and per-generator results are in the [unaltered aggregate JSON](evaluation-panels/image-fresh-v12-result.json).

## Pixel localization remains weak

The current selected masks reach the predeclared IoU-at-least-0.3 criterion on
only **3 of 48** edited images. Mean IoU across all 48, including misses, is
0.048467; mean pixel F1 is 0.073028. Two genuine images have selected pixels;
mean highlighted source area across all 48 genuine images is 0.3042%.

The candidate changes file-presence calibration, not pixel selection. A higher
file response must not be represented as better localization, a new edited-pixel
mask or independent corroboration. These results support developing better
source-bound features and ordinary-edit controls rather than further adjustment
of this candidate's frozen points. The 48 pairs are now consumed test data.

## Execution and separate replay

All 96 planned inputs completed without errors: 102 actual forwards including
four consumed-reference forwards and two new-input repeats. Retained-reference
head/map parity and gate-side checks pass at the fixed tolerance. Native scoring
took 75.53 seconds with no guard activation or sampling errors.

The [separate model-free audit](evaluation-panels/image-fresh-v12-source-audit.json)
replayed all 96 source/input/map/component bindings, independently indexed 384
source-quadrant means and independently calculated 192 fixed file probabilities.
The maximum probability difference is exactly zero. It loaded no weights and
ran no neural forwards. Its separate guard completed in 6.58 seconds with one
process-sampling disappearance and no guard activation.

The [execution receipt](evaluation-panels/image-fresh-v12-execution.json) preserves
sampled process-tree RSS lower bounds and private-record commitments. These are
not total GPU memory measurements or population accuracy claims. Fifty-five
selected regression tests pass. Outside human reviewers remain zero; an
independent arithmetic implementation is not an outside review.

No full-app promotion test was attempted because the prerequisite gates failed.
Serving models/calibration, USD9.99 prepayment, model-name filtering, existing
evidence exports and expert-engagement scope are unchanged. No customer upload,
external verifier call, new model download or GCP deployment occurred. Raw
photos, native maps, source pairings and per-input journals remain private.
