# Second generator: native-coordinate CPU qualification

Use the retained MIT LivePortrait human source/weights, verified by the exact
v4 asset catalogue and pinned revisions in the [rights review](av-transfer-rights-check-v130.md).
No InsightFace code/checkpoints, publisher landmark ONNX, animal models or new
weights are admitted. Use the actual successfully installed
[v132 isolated geometry runtime](portrait-geometry-runtime-v132-method.md), verifying
its original wheels and all installed source/native bytes before each execution.

Select the same deterministic first fitting-source native picture used by v118;
this is a consumed source for runtime qualification, **not fresh detector testing**.
From its bound picture bank choose the closest picture to 320 ms earlier, with
lower index as tie-break and at most 40 ms residual. Bind both PNG bytes, decoded
RGB and native source clocks before inference. Do not choose another source if
geometry or rendering fails. No source speech encoder runs in this stage.

Run CPU only, float32, one Torch/inter-op/OpenCV thread, deterministic algorithms
and seed 133. No MPS/CUDA fallback. MediaPipe uses IMAGE mode, CPU, up to two faces,
all three confidence settings 0.5, no blendshapes/transform output. Require exactly
one finite 478-point mesh per picture; call current, earlier, current and require
bitwise repeated geometry. Keep face points private. No identity recognition.

The original loader rejects unsafe checkpoint globals and uses `weights_only`
with strict state loading. Observe five strict loads and verify every loaded
tensor exactly against the checkpoint; require CPU float32 parameters. Hook each
of the five top-level generator modules. Render the current crop with its own
measured aperture, repeat, the earlier picture's aperture, and a fixed authored
aperture of 0.5. Require four inputs/twenty top-module forwards, byte-identical
repeat and a changed authored counterfactual. Report earlier-response magnitude
even if zero: a scalar mouth aperture does not encode phonemes or guarantee
that a generated mouth matches an audio time shift.

The generator returns 512-pixel crops from a 256-pixel input. Preserve pixel-centre
geometry: the source-to-output affine equals twice the source-to-input affine,
**plus 0.5 in each translation**. Invert that complete matrix to return to native
pixels. Composite only the recorded feathered source-mouth ellipse; require
changed pixels, and byte-identical original pixels outside support. Keep crops,
composites, exact changed masks, points, ratios and transforms private. Publish
only counts/timing and whole-record/freeze hashes. The three task API calls and
twenty hooked module forwards are separate counters; internal face-task graph
invocations are not independently instrumented.

Freeze and manually publish this tested contract before the first neural call.
Execute sequentially outside the sandbox with the qualified exact monitor:
900 seconds, 6 GiB initial available RAM and sampled descendant RSS, 3 GiB initial
project/archive disk, 1.5 GiB running RAM/disk reserves. Admission denial creates
no attempted stage and can be checked again. Once attempted, preserve every
failure/partial artifact; no retry, overwrite, source substitution or relaxed
limits. Samples are lower bounds rather than guaranteed resource peaks.

Success qualifies only the construction runtime/native mapping. It does not
prove realistic lip sync, correct phonetic timing, deepfake detection accuracy,
new-source transfer or customer confidence calibration. Those need separate
frozen evaluation. No application model/default, Actions, outreach, payment or
deployment changes occur here.
