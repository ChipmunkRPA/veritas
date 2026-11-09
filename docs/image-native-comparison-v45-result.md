# Native image comparison: a strong sensitivity gain and a useful agreement view

The trained strict localizer usefully locates **21/27 edits versus 4/27** for
the current native check on previously unused source groups. It adds 17 useful
locations and loses none of the current four. It also falsely flags more
original photographs: **9/27 versus 2/27**. Keep this as a sensitivity/error
tradeoff for confirmation, not an automatic rejection or a claim of readiness.

The actual intersection of the two methods retains all four useful current
locations while reducing original false flags from 2/27 to 0/27, tone flags
6/27 to 2/27, and blur flags 8/27 to 0/27. It supplies a complementary stricter
view; it does not find the extra 17 edits. These are two different useful
operating behaviors, not proof of AI authorship or calibrated confidence.

## Every fixed readout

The [prospectively published protocol](image-native-comparison-v45-protocol.md)
and [exact freeze](evaluation-panels/image-native-comparison-v45-freeze.json)
preceded every new source prediction. All methods use the same 27 source photos
in 24 connected components, with edited/original/tone/blur views. These groups
are now consumed comparison/selection data. All belong to the earlier editing
family; neither new-generator generalization nor raw-camera authenticity is
established.

Each count below is per photograph, out of 27 in that column. Correlated views
and photographs in a shared component are not independent people or captures.

| Method | Useful edit localizations | No region on edited photos | Original false flags | Tone false flags | Blur false flags |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current native check | 4 | 20 | 2 | 6 | 8 |
| Adapted, strict | **21** | 3 | 9 | 8 | 6 |
| Adapted, middle | 21 | 2 | 13 | 13 | 12 |
| Adapted, loose | 19 | 1 | 16 | 16 | 16 |
| FUSED native reference | 2 | 23 | 1 | 2 | 3 |
| Current OR strict | 21 | 3 | 11 | 11 | 12 |
| Current AND strict | 4 | 21 | **0** | **2** | **0** |

The actual OR mask adds no useful location beyond the strict adapted mask and
creates additional false flags. Middle/loose settings do not improve useful
localization here. FUSED's native-format reference has fewer false flags but
finds fewer edits; this does not invalidate its separate JPEG use or architecture.
All methods remain in the comparison package rather than being deleted.

The strict/middle/loose internal map settings are respectively
`p0.90-a0.01000`, `p0.60-a0.01000` and `p0.40-a0.00100`. They are fixed raw-map
parameters chosen from earlier calibration, **not customer confidence scores**.
Useful localization requires both source-mask IoU >=0.30 and recall >=0.50.
An absent region does not establish whole-file originality.

## Source-balanced results and uncertainty

The current useful fraction averaged equally across components is 13.9%;
strict adapted is 77.8%. The paired gain is 63.9 percentage points, with
conditional paired-group bootstrap 95% interval **[44.4, 83.3] percentage
points**. These are 10,000 fixed-seed, pointwise draws from the 24 consumed
components. The interval excludes fitting/calibration uncertainty, unknown
creator aliases, pretraining overlap and new-domain shifts. It is not a
simultaneous guarantee after comparing multiple methods or a customer interval.

Original/tone/blur false groups out of 24 are 2/6/8 current, 8/7/6 strict
adapted and 0/2/0 intersection. Any-control false groups are 8, 8 and 2,
respectively. The same any-control count conceals the strict model's additional
original-image false alarms, so the separate control types remain visible.

Average original-photo area wrongly highlighted rises from **0.302% to 1.772%**
with the strict model; its tone and blur averages fall from 2.002%/1.321% to
1.628%/0.931%. The intersection has zero original/blur wrong area and 0.675%
tone wrong area. Counts and area must be reported together.

The owner-directed tradeoff comparison keeps all methods and applies no
automatic per-control zero-increase rejection. On these observed axes, strict
adapted and the agreement intersection form the non-dominated frontier.
Point estimates are not population dominance or evidence of calibrated authorship.

## Actual execution and integrity

All **327 neural calls** completed: 324 unique predictions across three detector
paths plus three exact first-input repeats. All 15 producer phases, five
model-free readouts and the final model-free summary exited zero without a
guard, sampling error or recorded resource crossing. Raw maps passed exact
save/readback and input/geometry/hash binding. This is not a complete second
neural replay of all predictions.

All 540 standalone source-mask measurements matched the frozen production
reconstruction before 216 actual union/intersection masks were measured.
Separate contingency/group arithmetic reproduced every method's useful counts,
control flags and source-balanced useful fractions. All model assets, frozen
code and the completed learned checkpoint were hash-verified again. The
protected audio calibration is unchanged.

The [complete aggregate](evaluation-panels/image-native-comparison-v45-result.json)
contains every readout, error type, new/lost location count and paired interval.
The [execution receipt](evaluation-panels/image-native-comparison-v45-execution.json)
binds all terminal resource-profile hashes and observed durations. Resource
limits stayed 900 seconds/6 GiB sampled worker-tree RSS per phase with the
original RAM/disk reserves. Sampled RSS is not a guaranteed peak or complete
GPU-memory measurement. Forty-seven focused local checks and Ruff pass.

Raw sources, maps, creator identities, pairings and learned weights remain
private. Outside human reviewers: zero. No workflow, outreach, charge or GCP
resource was used. App defaults, customer reports and overall confidence are
unchanged by this research execution.

## Next comparison

Advance both the strict standalone localizer and the actual agreement view to
separately frozen confirmation, with the current check retained. Keep the
extra original-image false alarms explicit. Do not select new settings using
confirmation labels or assume that a low false-alarm agreement view detects
most edits. The diagnostic/confirmation source roles have not been forwarded
by this study. Later new-generator, natural-image and full application-flow
tests remain necessary before a new default or confidence contribution.
