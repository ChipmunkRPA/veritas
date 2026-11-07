# Patch-drift localizer: native contract before project images

This is a new regional-edit hypothesis, not a revived failed FUSED crop or
Smaug gate. No project image may be scored before its separate data/method freeze.
This first contract uses six authored synthetic RGB inputs and no reference masks.
It establishes implementation behavior, not detection quality.

## Sources and commercial fit

[TRAIL](https://github.com/VishalJ99/trail-image-edit-localization/tree/0fce0f1de2ac0d219baf91592ee2bca95ec0a219)
provides a fixed perturbation and corresponding-patch cosine drift. Its MIT grant
is retained in `docs/third-party/TRAIL-v8-MIT.txt`; four NumPy primitives are copied
unchanged with attribution. Only inspected local numerical source is used as a
reference. No remote loader, optional baseline, gated model or dataset script runs.

Use the native DINOv2 ViT-B/14 register backbone (86,583,552 parameters), whose
code and weights have the publisher's explicit
[Apache-2.0 grant](https://github.com/facebookresearch/dinov2/blob/b8931f7bf91576930313be2c6d6af376033b35f0/README.md#license).
The existing model manifest pins its346,393,545-byte checkpoint to
`73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71`.
The already verified architecture archive is pinned at `b8931f7bf91576930313be2c6d6af376033b35f0`.
No localization head is learned. No other RADAR/depth model runs; some such
dependencies are non-commercial and are outside this experiment.

The published smaller-model comparison selects register-base block10; fix that
readout here without searching project outcomes. This uses a different, explicitly
pinned architecture revision from TRAIL's `7764ea0...` manifest, not its7B headline
system. Published performance is motivation only. Full pretraining overlap and
source-dataset/personality rights remain unestablished.

## Fixed input and readout

- Source RGB is resized to448 square with Pillow bilinear resampling. Aspect-ratio
  distortion is explicit; source projection later uses both original dimensions.
- Apply global4-by-4 blockwise Haar-detail attenuation at alpha0.2: retain80% of
  detail. Keep perturbed values in float32; do not re-encode or round them to uint8.
- Normalize both views with ImageNet mean/std, float32 NCHW. Use the normal pinned
  model in eval mode with all parameters frozen; disable optional xFormers.
- Extract final-LayerNorm block10 patch tokens, excluding one CLS and four register
  tokens. Require shape2-by-1024-by-768. Corresponding-patch one-minus-cosine is
  computed in NumPy float32, then a reflect-padded3-by-3 median gives a32-square map.
  This CPU arithmetic adaptation is not asserted identical to every GPU reduction
  used by the external paper runner.
- Two later predeclared readouts are raw median drift and within-image median/IQR
  normalization, with float32 epsilon as the scale floor. Neither is an AI
  probability. Their own genuine-only operating points require a separate freeze.
- A future display encoding must preserve strict raw-score eligibility in the
  uint8 high bit. Magnitude is an inspection index relative to its operating point,
  not confidence. Existing reliability exclusions and region caps remain controls.

## Native checks fixed before execution

Six deterministic authored inputs cover gradients, checker patterns, colour fields,
stripes, noise and unequal source dimensions. Verify exact reference Haar/drift/
median arrays against the retained inspected MIT primitives. Compare manual
normalization with separate CPU Torch arithmetic within5e-7 maximum absolute error.
Compare actual native intermediate-layer features with a full-forward block hook
plus the model's final normalization and last1024 tokens; require exact equality.
Check full checkpoint state loading, architecture/counts, finite float32 arrays
and complete expected shapes. Any failure stops before project scoring; preserve
the failed attempt and do not relax these limits after observing it.

Freeze code, reference objects, assets and runtime before running this synthetic
contract. Native checks may read no project photo, customer upload or edit mask.
Retain private synthetic feature arrays and public hashes/aggregates. A bounded
parent records completion, sampled memory and observation failures, stopping above
6GiB RSS, below1.5GiB available RAM/disk or after30minutes. A sampled memory maximum
is a lower bound, not proof of peak resource use.

## Required next stage

After native success, freeze disjoint consumed genuine calibration sources and
development original/edit pairs before project forwards. Evaluate both PNG and
JPEG75, paired source groups, genuine highlights, missed regions and overlap under
the existing display caps. Advance only a qualified readout to separately frozen
fresh sources. No service, overall confidence or public threshold claim changes
follow from this synthetic check. Outside human reviewers remain zero; GCP stays last.
