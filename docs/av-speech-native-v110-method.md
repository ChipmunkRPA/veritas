# Native speech feature contract for timestamp-aware mouth conditioning

The [v109 clock plan](av-conditioning-clock-v109-result.md) establishes source
indices and explicit uncertainty. This next qualification checks actual CPU
encoder outputs and their routing before a new generator/source run. It does
not fit a detector, consume source authenticity responses or change serving.

## Exact permitted native path

Use the three cached Apache-2.0 speech assets from the retained v7 publisher
plan: encoder/decoder safetensors, model config and preprocessing config. Hash
all files against their publisher-bound records before execution. Read only
encoder tensors with the safe loader, load strictly, and compare every loaded
value with its source tensor. Do not instantiate a decoder, generator, VAE or
other model, download assets or enable remote code.

Use CPUfloat32, eager attention, inference mode, deterministic algorithms and
one computation/interoperability thread. Source inputs must be finite mono
16-kHz float32 arrays, 640 through 480,000 samples and within [-1,1]. The
extractor explicitly pads/truncates to 480,000 samples without normalization.
Require exact finite (1,80,3000) CPU log-mel input. Preserve the reviewed
whole-input normalization and unmasked attention; changing those would require
a different graph and comparison.

Require five finite (1,1500,384) native states. Independently capture the input
to all four encoder layers and compare the first four API states bitwise;
compare the last API state with the final encoder output. Pack time first,
then layer, yielding (1500,5,384). Missing, reordered or nonfinite arrays fail
the execution contract. They are not negative authenticity evidence.

## Predeclared authored experiment

No source file is used in this qualification. Make exactly three forwards:

1. Two seconds of fixed 440/880-Hz tones at the prescribed amplitudes.
2. A byte-identical copy, requiring exact input/state hashes and bitwise replay.
3. The same first two seconds in a four-second waveform, with a fixed quiet
   733-Hz suffix beginning at 2.4 seconds.

Using the first native hidden output, compare all 50 canonical 25-Hz frame
chunks from the new timestamp-aware array selector with the retained v7
tick/layer routing, requiring bitwise equality. This proves compatibility on
that authored input, not new-source or generator parity.

For the suffix probe, compare the first 100 speech positions layer by layer.
Retain changed-value counts and maximum absolute differences whether small,
large or zero. The first two seconds of raw input must remain identical and
the added suffix begins beyond their nominal convolution support. This probes
broader context dependence of otherwise identically indexed features. It is
not AI evidence, a padding-removal method or measured phonetic synchronization.
Do not select a different probe or gate from the observed differences.

## Freeze, resources and preservation

Freeze this procedure, core/runtime/tests, retained array/clock code, speech
asset identities, installed library source and package versions before neural
execution. An owned standard-library child/grandchild preflight uses the
unchanged shared v108 sampler in a new exclusive monitor directory. Its
reused receipt schema remains recorded; inspect actual exit/profile and
receipt hashes before admitting neural work. This is a new qualification,
not a retry of any historical failed audio attempt.

The authored stage retains 900 seconds, 6-GiB sampled descendant RSS, 6-GiB
initial available RAM, 3-GiB initial project/archive disk and 1.5-GiB running
RAM/disk reserves. Monitor-only work retains 900 seconds, 1.5-GiB sampled RSS,
3-GiB initial RAM/disk and 1.5-GiB reserves. Sample every .25 seconds, outside
the sandbox; samples are not guaranteed peaks. Read-only denied admission
creates no attempted model stage. Do not lower limits or terminate unrelated
apps. Model stages are sequential.

Use exclusive paths and keep all partial/failed attempts. No retry, overwrite,
silent device/model fallback or source substitution is permitted. Publish code,
aggregate authored measurements and whole-artifact hashes only. Native source
features and model weights remain private. The source-conditioned generation,
articulation detector, new calibration, fresh error evaluation and full customer
flow still need separate implementation/execution. No customer conclusion or
overall confidence can use this qualification alone. No outreach, Actions,
real payment or GCP step is part of it.
