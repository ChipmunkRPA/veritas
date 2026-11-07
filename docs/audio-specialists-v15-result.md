# Separately calibrated voice views: completed development result

None of the three alternatives passed the frozen improvement gates. Keep the
current serving audio model and its calibration. Adding the filtered-band view
did not improve file detection; background-specific calibration reduced false
flags while losing too many AI-containing files. No fresh source reserve was
used to rescue a failed method, and no overall customer confidence changed.

## Actual experiment and scope

All4,004 planned eight-second files completed their sequential native roles:
1,008 specialist-calibration,1,008 combination-fitting,1,008 combination-calibration
and980 reserved development inputs. Roles use143 consumed connected foreground
pairs, with35 in the reserved role, eight AI generators and48 role-disjoint
background originals. Overlapping four-second windows cover0–4,2–6 and4–8seconds.
Local labels follow the two exact64,000-sample foreground portions rather than
copying a whole-file AI label onto human portions.

Clean, noise, music and genuine overlapping speech conditions include±5dB
foreground/background ratios. Both views use the existing commercially reviewed
voice detector. The second view filters the whole eight-second waveform to
100–3,800Hz before extracting its source-clock windows. Original-window spectral
flatness chooses the predeclared concentrated/diffuse context. The two views and
overlapping windows are dependent; they are not independent votes.

Four separately calibrated variants were frozen, including all combined
artifacts and genuine-only internal operating points, before the final role's
neural inference. The reference is a **development calibration of the current
detector**, not the complete serving application's overall-confidence flow.
Minimum support stays25 connected groups per class and the reference prior50:50.
The internal comparison point is not a public screening threshold or accuracy
promise. All underlying sources were already consumed development data.

## File comparison

| Variant | AI found /735 | Genuine false flags /245 | Foregrounds with a false flag /35 | Inconclusive /980 | Source/class-balanced Brier |
|---|---:|---:|---:|---:|---:|
| Full waveform, pooled |579|10|7|42|0.134112|
| Full waveform, context-specific |520|5|3|42|0.125136|
| Full + filtered view, pooled |579|10|7|42|0.149235|
| Full + filtered view, context-specific |520|5|3|42|0.140638|

The filtered view changes some probabilities but none of the final internal
flags relative to its corresponding full-waveform method. Its Brier is worse.
Context-specific calibration halves genuine file false flags, but loses59
AI-containing files. A lower Brier is insufficient to accept this detection
tradeoff. Reused foregrounds/backgrounds make the many rendered files correlated;
245 genuine files are not245 independent original recordings.

## Strong background conditions

Each condition below has105 AI-containing and35 genuine files. Combined-view
methods have the same flag counts as their corresponding full-waveform method.

| Condition (foreground5dB below background) | Pooled AI found /105 | Context-specific AI found /105 | Pooled false flags /35 | Context-specific false flags /35 |
|---|---:|---:|---:|---:|
| Noise |72|68|1|1|
| Music |65|58|0|0|
| Genuine overlapping speech |85|71|3|1|

Across these conditions the pooled method finds222/315 AI-containing files;
the context-specific method finds197/315. Their unassessed files remain explicit
and never become originality verdicts. These small-panel counts do not establish
customer accuracy or performance on unseen generators, people or recordings.

## Native verification and reusable outputs

All four native role receipts and their resource profiles completed with exit0
and no guard activation. The unique scored view/window counts were4,476,4,428,
4,445 and4,309. Four view/context repeats per role matched exactly. The earlier
six-case native parity check also passed. One process-monitor sampling race was
recorded during fusion fitting; it was not an input or model inference failure.
Resource values are sampled process-tree RSS lower bounds, not certified peaks.

The separate local source reviewer verified every role receipt, exact fitted
artifact/point commitment, sample location, local label and original speech-gate
status, then reproduced every retained confirmation output exactly. That review
uses the retained source commitments/scalars; it is not an independent replay
of all source waveforms or an external human review.

The [full result](evaluation-panels/audio-specialists-v15-result.json),
[local source review](evaluation-panels/audio-specialists-v15-independent-local-source-review.json)
and [scalar replay release](evaluation-panels/audio-specialists-v15-release.json)
publish local sample coordinates, null statuses, scalar responses, frozen
parameters, internal points and source commitments. Raw audio, backgrounds,
voice representations and customer files remain private. Model-free replay can
refit each method and test reserved-response invariance and missing-check
withholding. A successful numerical replay is not fresh detection validation.

The actual local scalar refit compared20,972 non-null local/file probabilities
with maximum difference0. Its adverse check changed5,740 assessed reserved
units while preserving140 unavailable units, with unchanged fitted parameters.
Removing a required check from a previously assessed file withheld its result;
cross-role source lineage was refused. The
[actual local replay receipt](evaluation-panels/audio-specialists-v15-local-scalar-reproduction-v4.json)
records these checks. This does not turn the rejected candidates into useful
replacements.

The [actual Linux run](https://github.com/ChipmunkRPA/veritas/actions/runs/42423467035)
also archived44 exact dependencies from the published package commit and
refitted all four methods. It reproduced20,972 probabilities with maximum
difference3.3306690738754696e-16, including the same missing-evidence and
reserved-response adverse checks. The
[decoded workflow receipt](evaluation-panels/audio-specialists-v15-linux-review.json)
records that result. One successful audio replay does not resolve the separate
passage fitting failures below or establish portable training reliability.

The separate passage replay exposed occasional v2 fitting nonconvergence on
Linux. This audio package does not claim reliable cross-platform fitting from
a local pass; any Linux failure must stay visible. Serving inference continues
to use the unchanged already-fitted calibration. Future solver or scoped-view
changes need a separate prospective experiment, preserving these results.
