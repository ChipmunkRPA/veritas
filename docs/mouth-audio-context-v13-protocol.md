# Sound-conditioned mouth evidence: prospective development experiment

The newer visual-only temporal learners recover mouth edits but also flag many
ordinary clips. Test one controlled addition of actual source-audio/mouth
correspondence, using the same modest nonlinear learner as v12. Freeze and
publish all executable code and input commitments before extracting project
audio features. This is consumed-source development, not a fresh accuracy study.

## Fixed measurements and comparison

Use all 160 v11 fitting clips from sixteen actors, 24 ordinary point clips from
four actors and 88 consumed diagnostic clips from eight actors. Preserve all
6,373 windows, their native source regions, clocks, masks, labels and prior
49 measured visual variables. Actor roles stay disjoint. No actor, condition,
generator, true interval, mask or paired original enters inference variables.

Decode every exact prepared file's mono16k PCM with the existing decoder.
Require its complete sample count and float32 checksum to equal the source
receipt, a single contiguous timeline at zero and exactly 640 samples per
25 Hz picture. Verify actual picture PTS/dimensions with the existing probe.
Recompute lip aperture/width from every committed seven-point mesh trace;
native visual geometry/learned responses were audited previously and are
inherited here, not rerun. Missing audio, geometry or support fails the case;
it cannot become an original verdict or be replaced by a fabricated trace.

Compute log RMS independently in each non-overlapping 640-sample block, with
amplitude floor 1e-8. Per old window, measure the middle and its complete before/middle/after support. These overlapping
measurements are dependent, not independent votes. Both frame ranges extend
one frame before/after their respective center ranges. Intersect anchors with [8, n-8) to keep identical support for lags
-8, -4, 0, 4 and 8 frames. Require at least four anchors per part. Do not pad,
wrap or silently change the lag grid. Compute aperture/audio Pearson correlation
at each lag, best lag/8, peak-minus-zero correlation, unshifted audio and aperture
standard deviations, and fraction of usable correlations: ten variables per
part, twenty additions. Degenerate variance has correlation zero and an explicit
unusable fraction, not claimed evidence of synchronization. Ties prefer zero,
smaller magnitude, then earlier lag. These are descriptive measurements, not
measured lip-sync ground truth or AI confidence.

Compare the unchanged 49-variable v12 model and one 69-variable candidate.
Both use fixed v12 hyperparameters, source/class/condition balancing and every
fitting window. No sweep, calibration change, noise suppression, new checkpoint
or post-result routing. Refit the visual-only control and require parity with
its retained margins/point. Export all candidate trees as finite numeric JSON;
native/export margins must agree within 1e-10 on all planned windows.

Extract fitting and point inputs first. Write both complete heads, internal
points and a fit-completion hash before diagnostic audio extraction/scoring.
Choose each point only from all ordinary point windows using
nextafter(max(0, margins), +infinity). Keep the retained v8 absolute evidence
first, otherwise use the top candidate window under the unchanged ordering and
top-twelve retention. Report candidate-only and retained-v8 combinations.

## Evaluation and verification

Keep every original v9/v11 requirement against retained v8: six additional useful
locations, at least six first-family locations including two redraw and two
retime, at least thirteen second-family locations, at most one lost success,
at most one additional ordinary flag overall and per condition, spatial IoU
gain at least .02, retained temporal IoU, coverage and native/source parity.
Also require at least as many useful locations and no more ordinary flags than
the visual-only v12 control. Retain failures, zero-overlap misses and every
ordinary condition. Internal ranking points are not public thresholds.

A separate audit must decode all 272 source audio tracks again directly through
signed16 PCM, check source/PCM hashes and clocks, and independently reconstruct
all new block energies, lip apertures, lag support, features, tree margins,
points, selections, intervals and aggregate counts. It must not train a new
head, reuse the primary feature function as its reference, rerun a visual model
or claim to be an outside person's review. Authored numeric controls must
include known lag sign, zero variance, no circular wrap, exact samples, omitted
support, source-role leakage and altered tree/point/feature rejection.

CREMA-D retains its ODbL/DbCL attribution. Existing permitted geometry, visual
and generator assets keep their separate notices. New mathematics is independently
written with installed NumPy/scikit-learn; no external checkpoint or raw data
download. Run extraction, fitting, diagnostic and full audit sequentially with
900-second, 6 GiB sampled RSS and 1,536 MiB available-memory/disk guards. Keep
failed attempts and sampling errors; do not restart without a new prospective
amendment. Publish code, commitments and complete aggregates only; raw audio,
new audio traces and fitted private heads remain private.

An initial source preflight failed before a freeze or any audio measurement:
the historical whole-app guard includes three subsequently evolved serving
files (face-sequence adapter, video loaders and model manifest). Preserve that
failure and the old guard unchanged. The new study checks every other original
code pin, the complete original public/private visual capsule, source masks and
actor roles. It binds the current dependencies it actually executes in its own
prospective freeze. It does not execute the three evolved serving paths or
claim the old whole-app guard passed. The published v12 capsule separately
authenticates the retained visual-only head and results.

Even a development pass still requires unused actors/new sentences, multiple
generators, genuine noise/music/overlap, independent human testing and customer
timestamp/frame, PDF/JSON and prepaid-flow verification. Clean studio audio is
not a noisy field recording. This experiment changes no serving confidence, public threshold, billing,
rollback or GCP resources.
