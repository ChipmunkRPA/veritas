# CO-SPY local research implementation

`model.py` adapts the released `sd-v1_4` CO-SPY artifact CNN and fusion arithmetic
from [Megum1/CO-SPY](https://github.com/Megum1/CO-SPY/tree/310d048387edaf11f236ed1c7180d0e1e823250f),
revision `310d048387edaf11f236ed1c7180d0e1e823250f`, copyright 2025 Siyuan Cheng.
The unchanged upstream MIT licence is in `LICENSE`.

The adaptation removes training/augmentation, remote constructors, and permissive
weight loading. It retains the released PIL bilinear resize/center crop, semantic
normalization, 224-square bilinear resize without antialiasing, VAE posterior mean
reconstruction, residual `(input - reconstruction) / 7 * 100`, CNN, semantic-first
fusion order, and raw sigmoid. No latent scaling, feature normalization, input
remapping, or reconstruction clamp is added. Config and weight loading are local;
all keys, shapes, dtypes and finite values must match. Both SigLIP towers are
constructed and strictly loaded, while inference calls `encode_image(normalize=False)`.
The `vision_only=True` option is explicitly unsupported.

The model is an isolated candidate: it is not registered, calibrated, admitted,
or enabled in Veritas serving. A sigmoid output is a raw model score and does not
establish calibrated detection probability or accuracy. Numerical parity and
source-disjoint evaluation require separate recorded evidence.

Dependencies have separate terms. The embedded Stable Diffusion v1.4 VAE is subject
to **CreativeML OpenRAIL-M**, including its use restrictions; the system is not
MIT-only. The pinned SigLIP base from `timm/ViT-SO400M-14-SigLIP-384` is Apache-2.0.
Dependency source, weight, configuration and licence pins are recorded in
[`cospy-readiness-2026-09-19`](../../../../docs/v2/qa/cospy-readiness-2026-09-19/README.md).
No model weights are vendored here. Optional diffusers provisioning remains in an
isolated research overlay; this module does not modify installed dependencies.
