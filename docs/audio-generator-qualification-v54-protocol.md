# Offline speech generator qualification on authored prompts

This prerequisite supports the fresh multi-generator detector comparison. It
does not use a research recording or measure detection accuracy. Qualify the
small Kokoro CPU generation path, including full English preprocessing, before
assigning source roles or synthesizing the evaluation cohort. Other generators
and the common fresh comparison remain required work.

Use Kokoro 0.9.4, Misaki 0.9.4, Torch 2.13.0, NumPy 2.2.6 and Transformers 5.15.1
in a separate Python 3.12.6 environment. Retain the full hashed dependency lock,
English language package release and installation inventory. The publisher
labels Kokoro code/weights Apache-2.0; the language package is MIT and bundled
dependency notices remain attached to their distributions. Model files come
from the pinned publisher revision recorded in v53; verify actual model/profile
bytes against LFS SHA-256 and config against its Git blob before generation.
Keep large assets on the authorized external development drive.

Verify every installed file against its available wheel RECORD hash, then bind
the actual installed inventory and critical code before model calls. Use only
explicit local config, checkpoint and stock profile paths. Refuse unexpected
pickle globals; load tensors with weights_only. The upstream constructor has a
lenient loading branch: independently require exact module/key/shape/dtype/value
coverage after construction, allowing only a uniform documented module prefix.
Partial loading cannot qualify merely because construction returned.

Use the two authored English prompts in the executable freeze, fixed seed 5407,
CPU float32, two intra-op threads, one inter-op thread and deterministic
algorithms. Run full English G2P twice, refusing dropped spoken tokens, unknown
phonemes or inputs longer than 510 symbols. Perform three generator forwards
in prompt order 0,1,0, resetting the fixed seed before each. Require exact
float32 waveform repetition for prompt 0. Record native 24 kHz sample counts,
duration, finite mono float32 values, nonzero energy and PCM clipping burden.
Do not hide peaks through unrecorded normalization or claim human intelligibility
review. Block network connections during smoke and pass no app/account secrets
to the child. No detection forward, fitting, cloud resource or charge is allowed.

Preserve the failed external-cache package-resolution attempt and the failed
first English-package download. The unchanged dependency resolution succeeded
with a separate internal build cache, and the official release asset API
supplied the language wheel. No neural call occurred during those attempts.
The original cache and receipts remain. This qualification must not rewrite or
silently retry a failed model smoke. Any correction requires a successor run.

Publish and remotely verify the exact freeze before generation. Observe each
stage with the unchanged corrected process monitor: 900 seconds, sampled tree
RSS 6 GiB, RAM/internal/external disk reserves 1.5 GiB, initial RAM 6 GiB and disk
3 GiB. A timeout is not proof of process termination. Preserve failed/guarded
outputs; no implicit retry or overwrite. Successful authored generation only
qualifies this runtime path. It does not establish fresh source independence,
detector performance, a calibrated combined confidence, other generators or
the serving default. No workflow, outreach or deployment is permitted.
