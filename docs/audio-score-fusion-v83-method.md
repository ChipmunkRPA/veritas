# Candidate combined-confidence methods v83

This is a tested **candidate training library**, not a completed fresh-data
experiment, improved confidence result or serving change. Its numerical and
status tests use authored examples. Before using the new source panel for
detector calls, bind this exact implementation, extraction method, model/runtime
bytes, source/mixture manifests and evaluation procedure in a published study
freeze. Do not tune these methods on selection or confirmation outcomes.

Retain current and complementary component methods and their existing decision
combinations. Add two small score-fusion candidates rather than immediately
discarding either backbone:

1. **Global fusion.** Fit a monotone logistic combination of the original raw
   AI score and the frozen complementary score. The complement already uses
   the original raw score together with the separate speech representation;
   these are correlated inputs, not independent votes. Fitting their actual
   joint response is preferable to treating agreement as multiplied certainty.
2. **Measured-context fusion.** Split by median spectral flatness measured on
   assessed fitting audio. Fit one score combination in each resulting context,
   then calibrate each on its corresponding independent calibration sources.
   The measured property describes the input spectrum; it is not a verified
   noise label, speaker separation result or oracle for which model is correct.

Prediction receives component scores and, for the context method, measured
audio flatness. It does not receive truth, provider, condition, source ID, role
or known noise level as predictors. The cutoff comes only from fitting audio.
Both candidates use two clipped input logits, nonnegative bounded slopes and a
fixed small ridge penalty. Each supported branch then learns a positive-slope
logistic calibration map solely from `confidence_cal` sources. Fitting and
calibration connected groups and source IDs must be disjoint. A second
calibration call cannot silently refit an already calibrated artifact.

Balance truth classes, observed generator families, connected groups, sources
and assessed variants during fitting and calibration. This prevents repeating
the same recording under nine conditions from giving it nine independent-source
weights. It defines a **50% AI reference prior**, not the prevalence of AI among
customer uploads. Later evaluation must report ordinary counts as well as
weighted calibration metrics and class/provider/background strata. Shared
backgrounds and matched scripts still create dependence within each role.
Different variants of one source can fall in different measured contexts, so
the two branch calibrations may share source groups within the calibration
role. Their agreement must not be treated as independent evidence.

At least three connected groups of each truth class are required to estimate a
branch. That is a numerical support minimum, not a reliability qualification.
An unsupported fitting or calibration branch stays explicitly unassessed;
never borrow another branch's calibration or convert missing scores to zero.
Extraction failures remain failures. A skipped speech check supplies no original
verdict. No VAD, provider label, test answer or file metadata chooses a favorable
candidate at prediction time.

The result is a balanced-reference-prior score and a relative AI/original
tendency. It is not a population probability, statistical confidence interval,
court conclusion or universal accuracy claim. Do not turn the internal
comparison point into a public screening-threshold claim. Judge Brier/log loss,
ranking, empirical reliability, coverage, both wrong directions and operating
tradeoffs on separately assigned selection sources, then confirm the fixed
choice on the final role. A modest extra error does not automatically reject a
substantially more useful method. No old model or rollback is removed here.
