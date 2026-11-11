# Transactional installation of the unchanged v139 convolution arithmetic

This helper hardens installation and input guards; it does not change the frozen
v139 tiling plan, halo calculation, convolution operator or workspace estimate.
The v139 code, failed v133 run and successful v139 artifacts remain unchanged.
Actual pretrained qualification in a new context remains separate.

`install(models, observer, budget=WORKSPACE)` first validates every eligible
convolution before mutating any `forward`. Only ordinary Conv2d/Conv3d semantics,
eval mode, CPU float32 parameters/buffers and numeric zero padding are supported;
custom/replaced forwards and transposed convolutions fail explicitly. Installation
may occur while global grad is enabled because it performs metadata checks only.
Shared module objects are patched once. If any later assignment fails, restore
all attempted assignments, including one that mutated and then raised.

At every forward, before either the original small-workspace path or the tiled
path, revalidate the module and require disabled grad, disabled CPU autocast,
one corresponding strided CPU float32 input and matching parameter dimensions.
Unsupported training/device/precision/padding modes are errors, with no fallback,
automatic conversion, downsampling or device transfer. Both no-grad and inference
mode satisfy the disabled-grad requirement.

Module `__call__`, existing spectral-normalization hooks, parameters and buffers
are not replaced. Existing pre-hooks run normally before the common forward guard.
`restore(records)` reinstates the exact original instance `forward` entry, or
removes the temporary override when lookup originally used the class descriptor.
It does not overwrite parameter/buffer or hook state.

Authored tests cover later-invalid-module atomicity, assignment-after-mutation
rollback, fast-path autograd/training/device/dtype/autocast/padding rejection,
grouped/strided/dilated 2D/3D numerical parity, exact own repeats, unchanged state,
spectral-normalization hooks and shared-module restoration. These use small
synthetic tensors/weights, not pretrained networks, source media or detector
predictions. They do not establish full-model memory, generator quality, phonetic
alignment or detection accuracy. No default, Actions, outreach or deployment
changes are part of this helper.
