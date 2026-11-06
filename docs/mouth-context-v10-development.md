# Surrounding-footage evidence: fixed v10 development protocol

V8 remains the better research candidate (14/32 known edit locations, 0/56
ordinary flags). V9 source-paired learning lost that gain; a scalar-ranking
bound ruled out rescuing V9 with only a lower point at the allowed flag count.
V10 tests source-relative time context and calibration at the actual video
window unit. No weight training, new model/media downloads, serving/default,
global confidence, billing or GCP changes during research.

## Ordinary calibration clips and exact clocks

Keep the existing four point actors (1043,1075,1086,1083); neither the 16 fitting
actors nor the eight video-development actors can enter point selection. These
sources and point actors are consumed development, not fresh confirmation.
The production context preflight passed all four; the 63-frame lower bound was
a previous generation-recipe restriction, not production eligibility.

Construct exactly six variants per point source: original, local light1.12,
portrait-resampling, speech-renderer-resampling, local Gaussian blur sigma2,
and sigma4. No neural AI positives or their labels enter point selection.
Use the prior permitted MediaPipe VIDEO CPU geometry and portrait crop/mouth
ellipse for light/portrait-resample, and unchanged permitted YuNet landmark-box
geometry/mouth ellipse for the other three ordinary edits. For portrait
resampling, crop256, resize512 and inverse(matrix*2), matching V6/V4; no
generator forward. Use the centred 20-frame support and two-frame edge opacity.
Detect mesh in sequential original picture order for every retained frame;
this is a new full-video calibration construction, not byte identity with V6's
sparsely detected stills. Preserve masks, geometry and unchanged outside pixels.

The original picture starts at source .04s, audio at0. Trim decoded PCM to the
exact common picture span, resetting the new lossless clip clock to0. Never
pad/loop/synthesize audio or pictures. Original picture frame counts56/59/60/50
become55/58/59/50 complete audio/video frames, respectively; last source frames
with incomplete sound spans are explicitly excluded. Keep original PTS/time
base, PCM mapping, selected sample/frame ranges and full source byte hashes.
All24 planned FFV1 RGB/PCM clips must be available. Audit every frame, mask and
sample, all ordinary operations and context geometry before selecting points.
Existing component permission declarations and CREMA-D ODbL/DbCL terms remain;
no assertion of complete training rights or pretrained disjointness.

## Fixed statistic and candidate priority

Use the unchanged production tracked centres/native triplets and corrected V5
crop association. Compute V8 and V9 raw mouth-head margins with their exact
retained weights. Both share the frozen first ten small vision blocks, but this
research uses separate verified loads and makes no shared-runtime speed claim.
No original reference or generator identity is available to either at inference.

Keep V6 window enumeration: two prior/two following centre means, middle span
.52–2.6s, centre gaps<=.55s, triplet span<=.13s. Enumerate every eligible window
before selecting the top12; never re-rank only the old retained top12. Let M be
mean middle triplet margin, L/R the prior/following two-centre means. Define
**response = min(M-L,M-R) - .25*abs(L-R)**. The measured support is the native
mouth/lower-face chip, not causal pixel attribution. Rank by response descending,
then start/track; retain12 and all native margin/time/track inputs needed to
reconstruct the enumeration. Raw source-relative responses are not probabilities.

For each head separately, set its new point just above max(0, maximum ranked
window response across all24 ordinary calibration clips). Use the same window
statistic for calibration and development. Inference uses `response >= point`.
Do not transfer the former maximum-still point to relative-window responses.

The predeclared primary arm keeps every V8 absolute flag at its original
point/window. Only when V8 has no absolute flag, add the highest V9 relative
window if it reaches its own point. A secondary arm applies the identical rule
with V8 relative evidence. Both use the same full panel/gates. Preserve the
absolute flag as first evidence if both signals fire; report the signals
separately, without converting different margins to a probability. Selection
priority is primary if it passes, otherwise secondary if it passes; both are
reported. This development choice still requires a new frozen confirmation.
One pass must not hide a failed arm or the added head/runtime cost.

## Frozen comparison and gates

Freeze this design, construction/scoring/window code, source plan, both heads/
weights and V8/V9 receipts before any new head or relative response on
calibration/development clips. Score all24 calibration clips and all88 consumed
V4/V7 development clips with both heads, including failures/unassessed cases.
Replay current production windows; on development also verify every source/
time/alignment/chip/tensor binding and each head's raw margins against its own
immutable V8/V9 receipt (tolerance1e-7, exact hashes for native inputs). Preserve
all old outcomes. No source substitutions, density changes or rule searches.

Evaluate current cue, V8 absolute, each relative signal, both combined arms and
all conditions/families. A first flagged window requires temporal IoU>=.2;
representative-frame native mask overlap is zero when unflagged/outside edit.
Each combined arm must gain>=6 net locations over V8, locate>=6/16 first-family
edits including>=2 redraw and>=2 retime, retain>=13/16 second-family locations
and lose<=1 prior location, add<=1 ordinary flag in total and<=1 per condition,
improve mean spatial IoU across all32 edits by>=.02, retain mean temporal IoU
and assessed coverage, and complete all88 baseline/native/raw-margin replays.
All24 calibration cases must be assessed and source audited. Apply the same
requirements to both arms; no easier secondary subset.

Failure preserves V8 as research and consumes no fresh actors. A pass allows
only a separately frozen confirmation on unseen actors AND new sentences,
multiple generators, realistic compression/resolution/ordinary controls, then
outside review, source-linked UI/PDF/JSON/paid-flow integration and separate
overall-confidence admission before any serving replacement. GCP stays last.
