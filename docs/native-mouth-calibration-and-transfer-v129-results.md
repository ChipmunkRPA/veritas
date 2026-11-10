# Native mouth and speech comparison: calibration and locked results

The explicit interaction between mouth appearance and speech features improved
the finite-panel result over the new visual-only head. It is the most promising
candidate for the next transfer test. Plain concatenation raised false alarms
substantially. All three heads and the current timing check remain available;
no serving default or customer confidence calculation changed.

At the **predeclared primary internal research setting**, across both held-out
roles, interaction localized 9/40 constructed AI edits against 4/40 for visual-only,
with 9/80 genuine/ordinary-control false flags against 7/80. These totals combine
two separately reported locked roles; they are not a population accuracy claim.

| Candidate | Selection: edits localized /20 | Selection: false flags /40 | Confirmation: edits localized /20 | Confirmation: false flags /40 |
| --- | ---: | ---: | ---: | ---: |
| Visual only | 4 | 5 | 0 | 2 |
| Visual + speech concatenation | 13 | 27 | 6 | 13 |
| Explicit interaction | 7 | 8 | 2 | 1 |

All candidates had at least one complete scoring window on every ready clip;
this is not 100% frame coverage. Unsupported starts, ends and gaps are not scored.
Interaction still missed 13/20 edits
in selection and 18/20 in confirmation. It added seven localizations and lost
two relative to visual-only; it added seven genuine flags and removed five.
The net gain therefore hides useful complementary detections as well as errors.
Keep the visual head for comparison and potential separately calibrated fusion.
Do not automatically OR these flags and call that calibrated confidence.

The actual current correlation adapter produced a measured window on only 12/60
selection clips and 6/60 confirmation clips. It flagged none. Only 4/20 and 2/20
AI clips, respectively, were assessed by that comparator; the remaining 16 and
18 were unassessed, not successful negative detections. Its zero false flags
apply to only eight and four assessed genuine controls. It remains useful as
a separate timing diagnostic. This replay supplies common recorded dense native
face coordinates; it does not assert serving-tracker or whole-pipeline parity.

## Own calibrations and frozen settings

Each frozen head received its own point and maximum-window clip calibration:
six successful fits on ten source groups, only two full connected contexts.
These are balanced **study probabilities**, not customer/population probabilities.
Whole-clip separation remains weak. Confirmation class-balanced clip Brier was
0.2498 visual-only, 0.2461 concatenation and 0.2275 interaction; lower is better
within this benchmark, where an always-0.5 prediction scores 0.25. Point Brier
was 0.2273, 0.1599 and 0.1731, respectively. Good point scores do not establish
a reliable whole-file authenticity verdict.

All three internal settings were frozen before held-out responses. For interaction:

| Calibration source-group budget | Selection AI /20; false flags /40 | Confirmation AI /20; false flags /40 |
| --- | --- | --- |
| 0 | 3; 1 | 2; 1 |
| 1 | 7; 8 | 2; 1 |
| 2, predeclared primary | 7; 8 | 2; 1 |

These settings are reported for completeness. They are not a public screening
threshold, a promised error rate or a newly selected default. Do not retune a
point on these now-consumed roles and describe its result as fresh validation.

## Evidence scope and separate replay

Candidate input selection used every supported native centre. Windows average
three consecutive centres with five actual native-picture inputs. Saved records
retain exact source times, before/during/after indices, inspected mouth rectangles
and separately calibrated point/clip scores. Ground-truth masks describe known
construction changes; the inspected rectangle is not a predicted edited-pixel
mask. Cached speech features may depend on the whole input, so an indexed speech
interval does not establish strictly local causal evidence.

The source inventory was corrected **before any held-out worker**: fitting has
18 source groups/four connected contexts; calibration ten/two; selection ten/three;
confirmation ten/three. Actor, sentence and known ancestry connect contexts.
The unused v127 freeze remains intact; v128 executes its unchanged scoring code
in an explicit corrected context. No bootstrap confidence interval is claimed
from three connected contexts per held-out role. Source codes are not proof of
distinct people, unknown ancestry or pretraining independence.

The separate v129 audit replayed all 120 records, 7,818 eligible centres,
23,454 margins, 22,734 native windows, 1,080 research outcomes and 558 saved
lag correlations. Independent equations checked descriptor/projection/margin/
sigmoid values, native picture-cell time unions, main confusion counts and
unweighted clip Brier. It did not repeat the neural encoders, independently
judge unavailable scope, reconstruct spatial masks or involve outside reviewers.
Both guarded audits exited successfully with no resource or sampling failure.

## Next action and reproducibility

Carry interaction, visual-only and their frozen calibrations into new placement
and additional-generator tests. This bank uses one generator and centred edits;
it cannot yet establish those forms of transfer. Expand beyond these few repeated sentence
contexts. Test the complete local prepaid analysis/evidence/PDF/JSON path before
changing defaults. Treat modest false-alarm increases as explicit tradeoffs;
neither a perfect-detector requirement nor an automatic zero-extra-flag veto
was imposed here. Preserve all comparison artifacts and rollback models.

Code and 34 focused authored tests were checked locally. Calibration, both
held-out workers and both arithmetic audits completed under unchanged resource
limits, with no new neural forwards during these array/scoring stages. The
earlier extraction/encoder counts remain separate. All source identities, raw
media, individual fingerprints, feature vectors, weights and evidence traces
remain private on Seagate. Only methods, aggregates and whole-artifact commitments
are published by direct push. Actions stays disabled; no outreach, charge or GCP
deployment occurred.

The complete records are the [calibration result](evaluation-panels/native-mouth-calibration-result-v126.json),
[selection result](evaluation-panels/native-mouth-selection-result-v128.json),
[confirmation result](evaluation-panels/native-mouth-confirmation-result-v128.json),
[selection replay](evaluation-panels/native-mouth-selection-arithmetic-review-v129.json)
and [confirmation replay](evaluation-panels/native-mouth-confirmation-arithmetic-review-v129.json).
See the [component correction](native-mouth-component-evaluation-v128-method.md)
and [arithmetic audit procedure](native-mouth-arithmetic-review-v129-method.md)
for exact scope and preserved history.
