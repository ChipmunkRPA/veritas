# Actual supported image paths: development comparison v11

This is a consumed-source development experiment. It is not fresh validation,
and it cannot by itself change customer confidence or admit a serving model.
The earlier TIDAL-on-JPEG prototype is not the app's JPEG reference.

## Inputs, truth and actual reference

The unchanged image-v10 roles contain 240 TGIF original/edit pairs, 60 pairs
per role. Each becomes its original PNG and a deterministic JPEG quality75:
960 inputs total. Exact original RGB differences supply the local edit mask.
JPEG compression is an ordinary transformation, not additional AI truth.
Every input retains its byte hash, decoded RGB hash, source dimensions and
integer-coordinate quadrant boxes. Connected parents never cross roles.

The source preparation already verified 167,586,843 source pixels. Its plan
SHA256 is `47cb416891498c746c54df306a9eedba2e241491dec16099c6eb493ef26c6e0d`.
It alone did not authorize scoring: a separate full protocol freeze is required.

Actual byte signatures select the same paths as the supported app: PNG uses
TIDAL v30.1 (publisher Apache2 weights); JPEG uses FUSED OpenSDI (publisher MIT
weights) and its ConvNeXt backbone (Apache2). Installed weight hashes and the
existing commercial-license review are pinned. No new weights are downloaded.
The native JPEG network is a large backbone; this experiment does not describe
all project models as small. Dataset rights and model licensing are distinct.

The reference runs the actual `SpliceDetector.localise` / `FusedDetector.localise`
postprocessing with each bound native response. It uses the existing eligibility,
quantization, reliable-area mask, connected components, minimum area and cap.
Pixel overlap uses selected connected pixels, never filled bounding rectangles.
Zero reliable cells are unassessed for the current localizer, not an original
file verdict. This reference excludes other application evidence packets,
whole-generation checks and the full application headline.

## Four predeclared arms and calibration roles

1. Primary response only: PNG native fraction above0.70; JPEG whole-image head.
2. Primary plus the current integrity response.
3. Primary plus four source-bound quadrant means from the original raw map.
4. Primary, integrity and all four quadrants.

Whole responses have whole-image editing-presence truth. Quadrants have local
mask-intersection truth: unchanged areas in edited files are genuine locally.
Means accumulate in float64, matching the earlier source plan. No selected or
quantized map replaces the original raw response in calibration.

Each scalar readout gets its own monotone, source/class-balanced logistic
calibration. PNG and JPEG have exclusive dependency families and consequently
separate combined fits and combined calibration. Dependent views are not votes.
The unchanged v2 engine uses positive-logit maxima within a dependency family.
The new scoped-consistency v3 engine is a separate research proposal and is
**not** evaluated by this frozen four-arm experiment.

`specialist_cal` fits scalar heads; `fusion_fit` fits the combined response;
`fusion_cal` calibrates the combination and sets internal operating points.
All four artifacts and all eight per-format points must be frozen before
`confirmation` neural inference. Minimum support remains25 connected groups
per class; L2=.001; reference prior50:50. These scores are research estimates,
not customer-specific probabilities of truth or public accuracy promises.

Per-format points use the identical complete genuine fusion_cal cohort across
all arms. Match the current localizer's flagged source-group count, capped at
floor(0.10*n). Select the genuine group-maximum order statistic at rank n-budget;
flag only strictly above it. Require25 complete groups. Missing current or
candidate measurements stay missing; they never become zero. Confirmation,
AI calibration labels, generators and ROC curves cannot select points.

## Native checks, resource limits and retained failure

PNG runs CPU float32 with two threads. Before every role, the hash-first retained
map in each class must match a new native response bitwise, with an exact repeat.
All144 retained PNG maps may then be reused after byte/RGB/map binding checks.
The first new input in each class is repeated. All remaining maps are native.

JPEG runs MPS float32, batch1, with the app's own inference-only interpreter.
Before every role, the same two consumed hash-first genuine/AI JPEG references
must reproduce the retained whole-image head and source-projected map at
atol=rtol=1e-4. Projection is the original torch bilinear interpolation with
align_corners=False. Their original native512 maps were not retained; do not
claim native-map bitwise parity. The image-head gate side must match. Repeat
these probes and the first new input per class. No upstream executable code
or unlicensed optional weights are imported.

Run each role/format in a separate sequential process. Limit each to900seconds,
6GiB sampled process-tree RSS and at least1536MiB free disk/available memory.
Heavy scoring waits for the exact existing audio and video-preparation jobs.
A failed process/profile/journal stays retained; no implicit resume, dropped
case, tolerance change or automatic retry. Store maps privately as lossless
NumPy/gzip, not public media. Logs/receipts distinguish native forwards from
cached postprocessing requests.

## Evaluation and advancement

Report PNG/JPEG separately: all120 confirmation inputs per format, AI found,
false flags, inconclusive and exact-point ties. Include source/class-balanced
Brier, local whole/quadrant Brier, source-bound calibrated local outputs, and
current connected-mask overlap. Quadrant presence is not pixel authorship.
Compare against both the calibrated primary-only arm and actual current
component selector. A continuation candidate must have full calibrated
coverage, no detection loss or additional false flags relative to either,
and lower Brier than the calibrated primary. Actual rates need not equal the
calibration budget. An improvement on one format cannot rescue another.

Passing only qualifies further independent/fresh testing and a complete app
flow comparison; it never admits serving automatically. No held-out source
reserve, customer upload, payment, public threshold claim or infrastructure is
changed here. All underlying panels were previously consumed. An earlier144-case
cached preview included fitting data and has no qualification status. Its first
precision mismatch and the corrected float64 replay remain recorded separately.

The first complete freeze failed queue preflight before any image worker or
neural forward: in-memory arm tuples compared unequal to their JSON list
representation. The original freeze and all its pinned sources remain in
`freeze-attempt1.json` / `freeze-attempt1-source`. The repair normalizes only the
declaration to JSON lists, with an exact frozen-policy roundtrip regression.
The replacement freeze retains the same arms, roles, numerical method, native
parity limits and gates. No detection output selected this repair.
