# Native image candidate runs locally; detection comparison pending

7 November 2026. The [acquired permitted candidate](image-smaug-v7-preparation.md)
runs on CPU in the separate research environment. All624 base tensors and
two classifier tensors match every released value after conversion to
CPUfloat32:438,760,472 base values plus769 head values. The app environment and
serving configuration remain unchanged.

The native processor was tested with the complete local asset set. Its1,016-byte
chat template was obtained from the same pinned publisher revision and verified
against its Git blob before loading; no model weights or original acquisition
record changed. [Processor completion receipt](evaluation-panels/image-v7-smaug-processor-completion.json).
Processor/config/library construction succeeds offline with native standard
classes. No publisher Python module or custom remote model code is executed.

## Actual synthetic check

Three authored RGB fixtures produce nine fixed source-bound crops through the
previously verified preprocessing. The standard native pipeline yields exactly
identical repeated batched outputs. Single/batch maximum logit difference is
2.384185791015625e-7; resulting image probabilities are identical. Independent
float64 mean-logit/sigmoid arithmetic differs by at most1.0897694871503383e-8.
The prospective tolerances remain1e-4 for single/batch logits and1e-5 for
probabilities/reduction. The prototype and portable helper return identical
complete synthetic results. [Synthetic receipt](evaluation-panels/image-v7-smaug-synthetic-runtime.json).

The helper refuses simultaneous known model studies and preserves a new output
directory, original logs and failures. Its worker limit is6GiB RSS, minimum
system-available memory1.5GiB and180seconds. The public-helper run exits0 with
no guard trigger. Largest observed worker RSS is4.17GB, a sampled lower bound
that can miss between-sample peaks; this is not full-app or cloud sizing.
[Recorded execution](evaluation-panels/image-v7-smaug-runtime-profile.json).

Use a separate environment with the recorded library versions and obtain the
exact pinned model assets separately. The helper takes no project/customer image:

```sh
PYTHONPATH=backend python -m scripts.audit_image_smaug_runtime_v7 profile \
  /path/to/pinned-model /path/to/new-output
```

This checks full state equality, native pipeline execution and outer arithmetic;
it is not a separately implemented encoder, detection accuracy, calibration or
pixel-edit localization. The fixtures have no detection-quality labels. A
complete frozen comparison against the current image checks, with genuine and
known generated/edited controls, remains required before any new-source
confirmation or serving decision. Whole-image scores do not supply an exact
edited-pixel mask. Project-image scores remain zero, no existing calibration
is inherited, and no overall confidence, customer report, billing or GCP
configuration changes. Outside human reviewers remain zero.
