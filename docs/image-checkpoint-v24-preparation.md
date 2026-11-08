# Memory-saving preparation for the full regional learning probe

The full corrected v23 probe remains resource-stopped. This successor prepares
activation checkpointing and precise stop measurements without changing its
1024-pixel input, complete backbone, trainable parameters, loss or optimizer.
It does not establish a detection improvement or change serving confidence.

## What is checked locally

All32 small local tests pass: five recomputation tests,15 resource-limit tests
and12 comparison-receipt checks. The
[preparation package](evaluation-panels/image-checkpoint-v24-preparation.json)
binds exact implementation/test bytes, prior protected files and actual reports.

On a small authored12-block network, checkpointing preserves the output, every
trainable gradient and every Adam-updated parameter exactly. Parameter identities
and state-dictionary keys stay unchanged. Local/global keyword arguments remain
bound to their correct block. Inputs without gradients still train the selected
parameters, and inference without gradients follows the ordinary path.

The recorded small contract counts444,416 packed nonparameter tensor bytes on
the ordinary path and51,200 on the recomputed path. These are sums of saved
autograd payload sizes after excluding parameter storage, not unique physical
memory or a measured peak-RSS reduction. The full detector has not run in this
new study; its memory reduction and exact parity remain unverified.

An initial authored test had an invalid fixture assignment while constructing
a partial backbone. Four tests passed and that setup failed. Its source and XML
are retained. Fixing the fixture to use a proper truncated ModuleList allows
the intended partial-backbone refusal to be tested. The final32-test report has
no failures. No model outcome or scientific acceptance rule was used to tune it.

## Explicit full-model comparison, still pending

The new worker uses the unchanged strictly loaded native image graph and the
fixed-frequency selection from v23. The memory-saving arm wraps all12 backbone
blocks with non-reentrant activation checkpointing and preserves RNG state.
It re-executes block operations during backward rather than retaining all their
intermediate activations. No source crop, smaller model, reduced resolution,
quantization or changed trainable set is substituted.

The installed PyTorch2.13.0 source/signature was inspected. The accessible
[official checkpoint documentation](https://docs.pytorch.org/docs/2.15/checkpoint.html)
is version2.15 and supplies conceptual guidance, not a claim that our installed
runtime is2.15. The2.13 URL was inaccessible. Its warning about changed state
during recomputation is why exact outputs, gradients and updates must be checked.

Run baseline and checkpointed full-model arms separately and sequentially. Each
must finish four authored forwards, two backward/update steps, reproduce its
reset exactly, match canonical native inference and restore the starting model.
The comparator requires all246 selected gradient/update commitments, four native
patches, unchanged frozen parameters and identical complete geometry. It then
compares the full logit/gradient/update commitments across both arms. Incomplete
counters, changed source size, missing tensors or a non-repeating arm are refused.
Payload reduction is recorded separately from mathematical agreement.

Append-only worker events record the start/completion of forward, backward and
update operations. If interrupted, their completed-operation counters are lower
bounds; an unfinished operation is not inferred complete or absent. Old v22/v23
failures remain unchanged and are not retried by this preparation.

## Resource observations and actual preflight

The new monitor retains900seconds,6GiB sampled process-tree RSS and1536MiB
available RAM/disk reserves. Every sample records the observed values and each
specific crossed limit, including simultaneous violations. Live sampling failure
still stops execution. This corrects observability of future stops; it does not
retroactively identify the exact trigger of the old opaque stop.

Before a new full-model attempt, require6GiB available RAM and3GiB free disk.
The earlier full probe sampled about3.8GB RSS before its mathematical failure;
the extra starting headroom allows room above the unchanged running reserve.
This stricter initial policy affects only this authored comparison, not customer
analysis pricing or public detection thresholds. Existing output folders and
failed worker files cannot be silently overwritten or automatically retried.

An actual observation-only preflight records5,032,820,736 available RAM bytes
and1,889,316,864 free disk bytes. Both are below the new starting-headroom rules.
It reports those two reasons and starts **no child/model worker**. This is a
successful refusal, not a passing full-model test. The owner has already been
asked to free disk space; no additional permission or outreach was assumed.

The next neural step needs resources, complete paired-arm execution and actual
cross-arm parity. Connected-source supervised training, its own calibration,
quality comparisons, untouched confirmation and full app testing remain ahead.
No project image is inferred or fitted here. Model assets and serving defaults
remain unchanged. Outside human reviewers remain0, GCP is deferred, and GitHub
Actions stays disabled. Publish local results by direct push only.
