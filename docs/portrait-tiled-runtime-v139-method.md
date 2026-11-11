# Bounded CPU convolutions after the actual v133 memory stop

The v133 CPU model stage finally passed initial RAM admission, then exceeded
the unchanged 6-GiB sampled descendant RSS limit. It stopped after 18.48 seconds,
at 8,480,178,176 observed bytes. The first generated crop/native/mask PNGs exist,
but no completion receipt or complete counter journal exists; do not invent its
total neural calls or claim original qualification succeeded. Preserve its code,
freeze, resource/log files, failure and all partial outputs unchanged.

The retained human animation architecture contains a 7×7×7 3D mask convolution
over a 142-channel, 16×64×64 volume. A full unfolded-column float32 representation
would exceed 10 GiB. This is a diagnostic workspace estimate, not an observed
operator allocation: the original stopped worker did not log internal operators.
Some 2D layers can also create large temporary buffers. Source downscaling,
different weights/precision and a higher resource ceiling are not used here.

This new isolated v139 context changes only CPU spatial convolution execution.
For a Conv2d/Conv3d whose conservative input-channel × kernel-volume × full-output
estimate exceeds 128 MiB, pad the original input once, partition the output grid,
and call the same functional convolution on each exact input halo. Include stride,
dilation, groups and bias. Assemble the complete output **before** any existing
normalization, activation, residual or attention operation. Weights, shapes and
all original model forward boundaries remain unchanged. Inference/eval, CPU,
float32 and numeric zero padding only; unsupported modes fail explicitly.

Choose tile dimensions from rightmost to leftmost so each estimated column
workspace is at most 128 MiB; no output position is removed. This estimate is not
a guarantee of library memory or total RSS. The same 6-GiB/900-second guard and
1.5-GiB running reserves remain authoritative. Authored small operator checks
compare full/tiled convolutions at rtol/atol 2e-5, including seven-sized kernels,
padding halos, stride, dilation, grouping, exact own repeats and unchanged state.
They do not run pretrained networks or prove the full model is bounded.

Reuse the original q133 qualification code object, same first consumed fitting
source/pictures, exact meshes procedure, seed, five strict-loaded checkpoints,
CPU float32, four conditions, full 256-input/512-output dimensions and native
composite mask. Override only the process-local loader to install the explicit
convolution wrappers. The frozen upstream source and original qualifier stay
unchanged. Record flushed before/after top-module and tiled-operator events so a
new failure has observable partial progress. The original three task inputs and
twenty top-module forwards remain separate counters; internal graph/functional
tile calls are not conflated with those model-input counters.

Before execution, bind a whole private manifest of the original partial PNGs and
original failure artifacts. After the four new renders, independently compare
the first new full crop with the preserved original crop: **maximum difference
at most one RGB byte level and mean difference at most 0.01 byte levels**, fixed
now before the new response. Save the measured fidelity even if it fails.
Require the new own repeat to remain byte-identical and all original response,
strict-load, module-count, geometry and native containment checks to pass.
The partial old reference is not a completed old-run accuracy/repeat certificate.

Test/freeze/manually publish this context before its first pretrained forward.
Execute sequentially outside the sandbox, with unchanged initial available RAM
6 GiB, sampled RSS 6 GiB, 900 seconds, initial project/archive disk 3 GiB and running
RAM/disk reserves 1.5 GiB. Samples are lower bounds, not peak guarantees. Preserve
any failure; no retry, overwrite, fallback, source substitution or relaxed limit.
The stopped acquisition controller is terminal and must not be restarted; later
unattempted sources can be scheduled separately after inspecting this stage.

Publish code, aggregates and whole-artifact commitments, with no raw media,
individual fingerprints, motion/voice vectors or parameters. This is still a
construction-runtime test, not achieved speech timing, new-source accuracy,
calibration or customer confidence. No serving/default, Actions, outreach,
payment or deployment change.
