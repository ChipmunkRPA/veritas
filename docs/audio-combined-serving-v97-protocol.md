# Combined audio serving: inference extraction and numerical qualification

Retain the global score combination selected before the completed v84/v88 final
test. Export only its inference parameters and the frozen complementary head's
normalization/coefficients into a private local bundle. Bind their original
whole-artifact hashes and the published selection. Do not refit, select the
retrospectively better context arm, copy source identifiers into the bundle,
or publish learned arrays. The old production calibration remains unchanged.

The new app module has no research-script dependency. A caller supplies the
existing checked voice model, speech-activity model and deadline. It supports
only original finite float32 mono 16-kHz audio of exactly 64,000 samples.
Shorter/longer inputs remain unassessed; do not pad, crop, average windows or
assign a whole-file score. At least half of the full source window must be
detected speech, matching the incumbent gate. Invalid/overlapping activity
spans, component responses, feature clocks, deadlines and calibration bytes
fail without an original verdict.

The complementary encoder retains its previously pinned local weights and
license bytes, CPU float32 eager execution, and no preprocessing of original
PCM. It pools means and population standard deviations from layers 4/8/12 over
all 199 valid feature frames. The fixed head combines those features with the
raw voice logit. The selected two-score logistic combination then applies its
own calibration. These inputs are correlated; the result is a balanced-reference
tendency, not independent votes, population accuracy or a per-file interval.
Background authenticity and speaker separation are not established.

Before neural or end-to-end checks, replay **all 4,176 completed records** from
the 48 original stages. Recheck full/compact journal bindings and successful
original resource profiles. Require lossless float32 representation hashes,
199-frame clocks, original PCM correspondence and one complete source window.
Compare the new inference arithmetic with the untouched v83 prediction code
using the original recorded component response and selected fitted artifact.
Maximum absolute probability error is fixed at 1e-12; every tendency must agree.
Preserve all unassessed inputs as missing, never zero or original. This replay
is consumed-data serving parity, not an additional accuracy experiment.

Publish code, authored tests and the aggregate freeze before replay. Recheck
the remote freeze blob and record its verification locally. Numerical-only
stages load no Torch/model and use a 900-second, 1.5-GiB sampled process RSS
budget, 3-GiB initial RAM/disk and 1.5-GiB running reserves on both volumes.
Observe resources every 0.25 seconds; samples are not true peaks. Each stage has
an exclusive profile/output destination; retain failures without implicit retry
or overwrite. Original neural-study budgets remain unchanged.

The new runtime remains unselected. Actual model extraction parity, measured
gate correspondence, app registration/resource planning, longer-file window
scope, UI/PDF/JSON evidence and prepayment/rollback checks still need separate
qualification before any default change. No Actions run, outreach, live charge
or GCP action is authorized by this protocol.
