# Frozen-normalizer optimizer audit and unchanged video comparison

The v23 audit checks the new Newton optimizer on each original problem using
that problem's stored center and scale exactly. Host-recomputed normalization
is recorded only as a diagnostic. This isolates the optimizer without requiring
two platforms to recreate old floating-point normalization bit for bit.
The optimizer, its independent KKT limit, source data and video rules are unchanged.

All102 local problems passed:17 audio,32 image,16 video and37 passage fits. The
maximum independent KKT residual is below1e-9, and the greatest new objective
excess over the original is2.220446049250313e-16. The same source/class balance,
smoothing, clipping and L2 .001 apply. Old successful and failed runs remain
unchanged. This is numerical research, not a new media detection benchmark.

All15 video artifacts, internal points and evaluation outputs are exactly equal
to v22 on the Mac. Holding normalization fixed in the old-problem audit does
not change the new video-model fitter. Its controlled results remain:

| Method | AI found /80 | Genuine false flags /80 | Inconclusive /160 | Source/class-balanced Brier |
|---|---:|---:|---:|---:|
| Direct appearance calibration |63|2|5|0.103192|
| Positive appearance calibration |63|2|5|0.101693|
| Family maximum |18|3|9|0.201978|
| Same maximum repeated with matched penalty |18|3|9|0.201978|
| Preserved per-view maxima |63|3|9|0.096880|
| Existing raw appearance cue |62|1|5|—|
| Direct calibration at mapped existing raw point |62|1|5|0.103192|

The matched-dimension control differs from the one-feature family maximum by
at most1.6653345369377348e-16 in probability. Preserving which check generated
the signal improves this combination, but none of the new arms beats the
stronger existing raw appearance cue on all frozen continuation requirements.
Keep customer defaults unchanged and fresh reserves unscored. A mapped raw
inspection point is not a new public confidence/direction threshold.

Every unavailable input stays explicit and source frame/time positions are
preserved. The adversarial reserved-response and missing-check tests pass. No
new localizer, pixel-authorship, identity or lip-sync claim is made. Source
clips and older reserved outcomes were consumed before this development study;
new genuine edit controls, fresh data and full-app evidence checks remain needed.

The [v22 Linux run](https://github.com/ChipmunkRPA/veritas/actions/runs/42436036702)
failed all three replicas at its exact host/stored normalizer comparison. Its
[failure receipt and artifact identities](evaluation-panels/preserved-evidence-v22-linux-failed-review.json)
remain public. The v23 audit revision uses the stored normalizer as part of the
problem itself; it does not weaken the optimizer's independent check or retry
the failed jobs. The [new review plan](preserved-evidence-v23-review-plan.md)
declares three separate v23 jobs and retains every result.

The [public v23 release](evaluation-panels/preserved-evidence-v23-release.json)
contains only reviewed scalar data, source coordinates, fitted parameters,
outcomes and code. No raw clips, face/voice features, alignments or customer
uploads are added. Local numerical replay and the actual immutable Linux
receipts are reported separately. Outside human reviewers remain0; numerical
portability does not establish fresh detection quality or court suitability.
