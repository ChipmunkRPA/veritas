# Reserved regional-image comparison rules, v27

These rules are prospective: v26 fitting is running, but no reserved source has
been inferred. This freezes the statistical readout and selection rules, not
the as-yet-unfinished final checkpoint or an inference execution. The final
checkpoint, inference producer, resource wrapper and geometry tests must be
bound in a separate published supplement before any reserved neural inference.

## Cohorts and scope

Use the exact v26 component roles. All photos remain in the denominator, with
four source-bound views per photo: the SD2 splice, original source, ordinary
tone retouch and ordinary blur retouch. Evaluate native and JPEG75 separately.
Compression changes the input, never the precompression AI truth coordinates.
Controls have an all-zero AI truth mask. These are labeled source-relative
controls; their source photography is not independently certified authentic.

Calibration has 32 groups/39 photos. Point estimation has 24/27. Diagnostics
have 48/59. Confirmation has 48/59. Related photos stay together throughout.
No failed, missing, skipped, unavailable or inconclusive check becomes a clear
observation or an original-file verdict. Report these statuses and their full
denominators. Any such failure prevents complete-cohort calibration/comparison;
retain its records instead of choosing a policy on the remaining successful
subset. Failure is not evidence against AI editing.

## Matched map pipeline and calibration

The baseline is the actual current native splice check for native inputs and
the actual current FUSED check for JPEG75 inputs, including their existing
image eligibility rules. Do not reuse predictions from earlier source sets.

Candidate maps use the full native model reconstruction and exact source-center
readout from v18; no crops, sharpening, rescoring or inverse-JPEG truth masks.
Both arms use the existing screening grid, 8-bit rounding, reliability conditions,
connected components, area constraints and per-map display caps implemented in
the frozen score-map/production-grid helpers. Score retained memberships against
the entire source mask, including truth in excluded conditions. This is one
map's measurement pipeline, not complete app packet admission or cross-map
competition; passing it does not establish end-to-end product performance.

The candidate's internal pixel grid is 0.30 through 0.90 in increments of 0.10.
Its minimum active source fraction is one of 0.0003, 0.001, 0.003 or 0.01.
Eligibility counts unquantized source responses strictly greater than the pixel
setting, requiring at least the ceiling of fraction times source pixel count.
Component thresholding uses the existing greater-or-equal comparison on the
recorded grid. These are 28 predeclared research policies, not public confidence
thresholds or probabilities of originality.

Choose one policy per format, using calibration data alone. A component is a
false-positive group for a negative kind if any of its photos has a retained
region for that kind. An eligible policy must have no more such groups than
the baseline, separately for originals, tone and blur. Maximize the mean over
components of each component's fraction of usefully localized positive photos.
Useful means IoU at least 0.30 and recall at least 0.50, as in the prior regional
studies. Tie-break by higher component-mean IoU, then higher pixel setting,
then higher minimum area. If none is eligible, record no selected policy;
do not introduce an all-off rule or retune the grid.

## Reserved comparisons and uncertainty

Apply the selected per-format policy unchanged. Point-stage progression requires
a strictly positive mean paired useful-localization gain and no observed increase
in false-positive groups for each of the three negative kinds. Diagnostics are
descriptive and never select an epoch, weight, response policy or confidence fit.
Only a format passing the point stage can advance to confirmation. Scoring any
reserved source consumes that source across all formats for freshness claims.

For paired localization differences, average photos within their fixed component,
then resample complete component differences with replacement. Use NumPy's
default generator, seed 20261108, 10,000 replicates and linear 2.5/97.5 percentiles.
Do not bootstrap individual photos as independent observations.

For the combined negative-group outcome, a component is flagged if any original,
tone or blur control is flagged. Count candidate-only false groups and
baseline-only false groups. An upper bound on the paired false-group-rate
difference is the 97.5% one-sided exact binomial upper bound for candidate-only
events minus the 97.5% one-sided exact lower bound for recovered events. A union
bound gives at least 95% pointwise coverage for that difference under independent
component sampling. Exact bounds use Clopper–Pearson binomial-CDF inversion;
the implementation is checked against the
[official SciPy exact-interval reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html).

Confirmation progression requires a positive lower localization bootstrap bound,
no observed false-group increase in each negative kind, and the false-group-rate
difference upper bound no greater than **0.05**. This five-percentage-point
research margin is fixed before observing reserved responses. It is not a SaaS
confidence threshold. Intervals are pointwise per format, not simultaneous across
formats. Approximate source groups, convenience sampling and unresolved owners
limit their population interpretation.

Zero observed false alarms do not remove rare-error uncertainty: with 48 groups
and no discordant events, this bound exceeds the margin. Such a result cannot
pass solely because its observed false-alarm difference is zero. If more evidence
is needed, prospectively plan a new independent cohort; never keep adding
confirmation samples until the same test passes or widen the margin afterward.

## Release boundary

A progression result is permission to continue research, not serving admission.
No probability calibration has occurred here; localizer sigmoid values are not
probabilities of authenticity, authorship or successful localization. Do not
attach them to the overall confidence score. Broader fresh commercially permitted
generators, independent testing and the full app evidence/export flow remain
necessary. Any later probability calibration needs separate source roles and
a prospectively frozen method. No outside reviewer has reviewed this work.
