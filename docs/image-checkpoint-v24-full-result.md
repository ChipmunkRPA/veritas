# Full authored training checks pass with lower sampled memory

The complete1024 native image graph passes both baseline and checkpointed
authored checks. Every output/logit, selected gradient and Adam-updated parameter
commitment agrees exactly across arms. Each arm also reproduces its reset and
restores the starting tensors. This qualifies the memory-saving research path
for a separately frozen source-learning experiment; it does not demonstrate
better detection accuracy or change serving confidence.

The [actual aggregate result](evaluation-panels/image-checkpoint-v24-full-result.json)
binds both probes, their resource profiles and the comparison. The unchanged
[preparation](image-checkpoint-v24-preparation.md) was published and remotely
verified before these new full-model phases. Both run after the owner-authorized,
copy-verified move of development files to external storage restored headroom.

| Actual local measure | Baseline | Checkpointed |
| --- | ---: | ---: |
| Authored complete model forwards | 4 | 4 |
| Backward/update steps | 2 | 2 |
| Reset/logit/gradient/update agreement | Exact | Exact |
| Packed nonparameter tensor bytes, each repeat | 2,843,287,688 | 263,745,032 |
| Sampled process-tree RSS maximum | 4,099,899,392 bytes | 1,748,533,248 bytes |
| Guarded duration | 15.340s | 16.685s |
| Guard/sampling error | None | None |

The observed sampled RSS maximum is about57% lower. Payload counts exclude
parameter storage but are not unique physical allocations. RSS is a sampled
lower bound, not a guarantee of the true instantaneous peak. These are one
sequential local pair with the same full model, spatial input, fixed-frequency
selection, CPUfloat32/two-thread execution and optimizer; not a general hardware
benchmark. Checkpointing performs extra block recomputation during backward,
which is recorded separately from complete model-forward counts.

No crop, smaller model, quantized weights, changed loss or reduced trainable
set was used to obtain the lower memory result. The unused/fixed parameters
remain frozen. Each arm records complete246-tensor selected gradients and
updates, native four-patch input correspondence and unchanged frozen weights.
The eight complete network forwards use only the authored geometric target,
not real/AI examples or any of the397 project image pairs.

The original v22 gradient-repeat failure and the v23 resource stop stay retained
and failed. This is a new successful comparison, not a reclassification of those
attempts. The earlier acquisition/reviewer failures also remain unchanged.

Next, freeze connected-source roles and the complete supervised learning,
calibration, comparison and confirmation protocol. Use the updated
[legacy-owner overlap audit](image-legacy-authors-v25-result.md) when selecting
roles; account for unresolved source identity. Real detection quality,
unseen-generator transfer and the full app flow must still be tested before
adoption. The current serving detectors and protected voice calibration are
unchanged. No outreach was sent, outside human reviewers remain0, GCP is deferred
and GitHub Actions remains disabled.
