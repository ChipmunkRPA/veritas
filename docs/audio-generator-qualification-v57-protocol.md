# Generator qualification successor: explicit neutral export state

Preserve the [v55 failure](audio-generator-qualification-v55-result.md), which
stopped before G2P or generation. The read-only constructor inspection found
548 saved tensors and 140 extra affine normalization parameters created by the
upstream AdaIN export workaround. No detector response or benchmark label was
used to identify this difference.

Use the same installed runtime/assets, authored texts, seed 5407, CPU float32,
thread settings, full G2P and three generator calls fixed in v54/v55. Keep the
corrected installed-package boundary and imported module-origin checks. This
successor verifies every saved checkpoint key, shape, dtype and actual loaded
value. Uniform module prefixes are supported; mixed prefixes and unexpected
saved keys are refused. No learned checkpoint tensor may be absent.

Permit exactly 140 additional tensors only as complete unsaved weight/bias
pairs in the pinned kokoro.istftnet.AdaIN1d's InstanceNorm1d modules. Require
affine mode, no running statistics, CPU float32, exact one scales and exact zero
biases. Reject every other missing state or nonneutral value. Freeze all model
parameters for inference and verify the complete state fingerprint remains
unchanged after all calls. An independent functional test compares affine
identity normalization to non-affine normalization exactly on authored input.
This checks the upstream graph as supplied; no weight or forward implementation
is modified, and incomplete learned weights are not waived.

All other prospective method and resource requirements remain as in v55:
publish/verify code/runtime/assets before generation; 900 seconds per stage,
sampled tree RSS 6 GiB, 1.5 GiB RAM/internal/external disk reserves, initial RAM
6 GiB and disk 3 GiB. Keep exact seeded waveform repetition, native 24 kHz
clock, finite mono float32 output, peak/clipping measurements, no dropped or
unknown/truncated spoken tokens, network blocking and no app secrets. Permit
two authored G2P calls and three generation calls, zero detector forwards.

Keep previous failed preparation/smoke/diagnostic launch attempts intact. No
implicit retry or overwrite. This is a separately named integrity correction,
not a retrospective pass for v55 or fresh detection validation. It qualifies
neither other generators nor a serving model/calibration. No workflow, outreach,
charge, deployment or default change is allowed.
