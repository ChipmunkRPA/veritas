# Prospective fixed-layer audio representation comparison

Freeze before project audio forwards or fitting. This follows the rejected
[full-vector comparison](audio-full-representation-v11-development-evaluation.md),
which recovered genuine-speech conclusions but still lost masked synthetic
speech. Merely removing guards cannot restore six generator families' required
AI retention. Different network layers may preserve acoustic details suppressed
by the pretrained fixed layer mixture; that is a hypothesis, not a known cause.

## Fixed arms and extraction

Keep every parameter of the existing commercial-permission speech model frozen,
including its original layer mixture and classifier. Keep exact waveforms,
preprocessing, attention masks, speech activity/windows and CPU float32 runtime.

- `final_control`: original unprojected, final pooled 1,024-vector.
- `layers`: concatenate masked means from hidden-state indices 8, 16 and 24,
  each 1,024-wide, within the original 25-state output (indices include the
  initial state). These thirds are selected before any new forward; no layer search.

An output hook reads the three layers without returning a replacement output or
mutating original states. Use the exact original backbone attention mask and
retain its digest, frame count, valid count and exact float32 feature digest.
No denoising, separation, waveform changes or new pretrained weights. The final
control isolates the new representation from removal of the old head-direction
projection. The exact v11 head/outcomes remain a fixed additional reference.

Every original extraction must reproduce the stored status, raw probability,
final-vector digest, VAD spans and complete four-second window exactly. Actual
source bytes and production PCM are checked at each input. Unassessed sources
produce no neural/layer measurements; never substitute zero authenticity.
A mismatch is retained as failed and stops extraction before fitting. Do not
silently erase a failed prefix or refit from an incomplete matrix. A resume needs
the exact header/input prefix and complete validation of existing records.

## Sources and learner

Same corrected v9 4,248 consumed inputs: 1,926 fit, 873 calibration, 1,449
development; 214/97/161 originals in 169/82/106 canonical connected groups.
All 62 unassessed original inputs remain unassessed. Noise/music/genuine-speech
backgrounds remain role-separated and variants stay with their parent source.
Known genuine/cloned speaker links remain intact; unknown biometric and
pretrained-data overlap remain unverified. Nine variants are correlated, not
nine independent users. Per-source CC-BY-4.0/MIT declarations and attributions
remain those in the original catalogue; do not imply a new blanket data license.

Both arms use the same original fitting-only weighted binary-log-loss objective:
equal truth/family/group/source/assessed-variant weights summing to one,
class-specific target smoothing from connected fitting-group counts, raw clipped
logit, chosen full representation and intercept. Weighted center/diagonal
standard deviations from fitting only, floor 1e-6, with sqrt(16/d) scaling and
total variance budget at most sixteen. No direction projection in either arm.
Ridge 0.001 on raw coefficient, 0.05 on representation coefficients, zero on
intercept. Explicit raw coefficient nonnegative; overlapping representation
coefficients remain unconstrained. No physical waveform-monotonicity claim.
Initialize raw coefficient one, others zero. L-BFGS-B, analytic gradient,
ftol 1e-12, gtol 1e-9, maxiter 1000; no tuning/early stopping. Failure to converge
fails the attempt. Metadata is not a predictor; use it only for weights/audits.

Complete all source records with zero extraction failures before fitting. Save
both fitted objects before independent calibration. Each arm receives its own
unchanged worst nonconformity per calibration group/truth, retaining missing
variants and requiring at least nine groups/class. Paired truth classes may
share a connected group within one role. Keep original internal 0.8/0.2 points,
alpha 0.1 guards and both raw/mapped six-decimal logit-change limits ln2. These
are internal benchmark rules, not a public SaaS threshold/accuracy promise.

## Complete development and decision

Compare all 1,449 inputs against the exact current guarded-service cache. Do
not change the service, v11 control, source matrix or any of sixteen gates.
Report both wrong-call directions, correct AI/original conclusions, AI without
an AI conclusion, abstention and unassessed/failure rates, balanced Brier/log loss,
AUC, every generator/condition/background and conditional connected-group paired
bootstrap intervals with original 2,000 draws/seed. More abstention is not a gain.

If either arm proposes original, run the unchanged filtered waveform through
the same original model, newly capturing layer means for both arms. Existing
450 final-vector probes are parity references, not substitutes for missing
layer data. Verify any available original raw/final-vector/filtered-PCM reference.
Record all failures as inconclusive; preserve them in planned denominators.

Only arms passing every unchanged service-baseline gate can advance. If just one
passes, select it. If both pass, prefer the simpler final control unless layers
add at least five correct conclusions, neither false-call direction is higher
and Brier is no worse. If neither passes, reject both without reserved scoring.
No post-result adjustment to penalties, layer choice, operating points or gates.

Before any confirmation, independently reconstruct all predictions, calibration
ranks, both stability decisions and complete statistical summaries; reproduce
fitting-only parameters. Re-run the fixed 99 native development cells and every
computed new probe, including original/layer/mask and mapped-output parity.
Additional probe cases audit provenance, not a fresh accuracy panel. Publish
safe parameters/outcomes for outside reproduction, excluding private/customer
audio, transcripts, provider IDs, local paths and full pretrained checkpoints.

The fixed 1,080 reserved inputs remain unchanged/unscored until development and
all numerical/native audits pass. A separately frozen complete confirmation,
paid-flow/evidence/PDF/JSON filtering and rollback checks precede any enablement.
Short-window results alone do not admit whole-recording/video confidence or
establish universal accuracy. No serving, confidence, payment/auth or GCP change.
Image/video/writing and real outside human testing remain separate open work.
