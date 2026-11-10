# Native mouth-render response qualification

Use the completed [native input preparation](native-mouth-inputs-v116-method.md)
and actual qualified speech states to test the retained commercial-permitted
mouth renderer. This is a one-source engineering contract before generating
the wider labelled panel. It is not a detector comparison, a lip-sync accuracy
test or evidence that requested audio conditioning achieved correct phonemes.

## Deterministic source and conditioning

Take the first original-order v116 fitting source with a planned native
20-picture edit. Use its central edit picture and exact v109 conditioning
plan. Require full retained-prefix support. Do not choose a source from detector
responses, replace a failed face, consume selection/confirmation outputs or
reinterpret old normalized-clock clips as native sources. Bind the chosen
picture, private states, source clock and successful source binding in an
exclusive private selection file before running a model.

Make a second conditioning plan exactly 320 ms earlier in source-audio time,
using rational subtraction and the original audio origin, sample count and
uncertainty. Retain the picture index as the fixed image being rendered. This
is a counterfactual audio context, not another picture's timestamp. Require
full earlier support; do not use a fixed number of native video frames as a
surrogate for elapsed audio time. Select ten ticks and five states through
the already qualified array helper, producing 50 by 384 CPUfloat32 values.

Run one cached MIT YuNet detection on the source picture with minimum face
size 48; require exactly one eligible face. Use the retained v7 landmark-box
crop and elliptical mouth compositing helper unchanged. The five-point crop
is approximate geometry, not measured 68-point facial anatomy.

## Fixed neural experiment

Use only the seven retained publisher-bound assets: MIT MuseTalk, MIT VAE and
Apache-2.0 speech components. Verify their actual bytes against the v7 manifest
and acquisition receipt before deserialization. Use the unchanged v7 safe,
strict loader and render implementation. No downloaded upstream Python is
executed, new model downloaded or noncommercial component introduced.

Require explicit MPS, float32, deterministic algorithms, one CPU thread and
the retained 0.65 per-process MPS fraction. No CPU fallback. The loader includes
the speech encoder, but this test uses its already qualified saved CPU states
and performs zero new speech-encoder forwards. On the same crop with the same
seed 117, render aligned conditioning, a byte-identical repeat, the 320-ms
earlier conditioning and all-zero conditioning, in that order.

Require identical repeated crop bytes and model proofs. Both changed conditions
must change at least one output crop pixel. Inverse-project each output and
composite through the measured source mouth alpha. Require a nonempty native
edit and zero changes where alpha is zero. Retain private generated crops,
native images, exact masks and proofs. Changed pixels demonstrate response to
conditioning, not naturalness, intended phonemes or achieved synchronization.
Do not tune a pixel threshold from the result.

Count actual module entries with forward-pre-hooks: eight VAE encoder calls,
four VAE decoder calls, four UNet calls and zero speech encoder calls. Together
with the single face detection, the fixed experiment requires 17 new neural
inputs. A partial failed stage has an unknown forward total unless separately
established; a missing receipt is not zero work. Record post-render MPS driver
allocation separately; neither it nor periodic descendant RSS is a true peak.

## Freeze, execution and interpretation

Freeze code/helpers/tests, this procedure, native input commitments, private
selection hash, permitted model assets and installed package versions before
execution. Reuse the exact v108 resource runner code object with isolated paths,
verification and worker routing. Upstream verification requires the exact
successful v110 owned-descendant preflight and authored contract. The reused
`authored` command and receipt schema describe execution routing, not a claim
that this actual source picture is artificial.

Keep the original 900-second, 6-GiB sampled descendant RSS, 6-GiB initial RAM,
3-GiB initial project/archive disk and 1.5-GiB running reserve policy. Model
execution is sequential, outside the sandbox. Denied read-only RAM admission
creates no attempted model-stage directory. Preserve all partial/failed stages;
no automatic retry, overwrite, silent device/model substitution or relaxed limit.

Publish aggregate outcomes and whole-artifact hashes only, without source IDs,
frames or individual face/voice fingerprints. Even a successful result qualifies
only this native renderer's response on one fitting source. Full native-clip
generation, quality/alignment evaluation, new articulation features and a
source-connected detector comparison remain required. No serving confidence,
calibration, report, rollback, real charge, outreach, Actions or GCP change.
