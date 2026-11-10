# Isolated runtime context for the native mouth-render contract

The first v117 **preparation** used the application environment, which does not
include `diffusers`. Its freeze command stopped with `PackageNotFoundError`
after writing the private deterministic selection. It made zero neural calls,
published no freeze and created no model-stage output. Preserve that selection
and the explicit failure record unchanged. This was not a model failure or
detector outcome.

The generator dependencies already exist in the isolated v7 research runtime.
Use that retained overlay explicitly, with the same application interpreter;
do not install or upgrade anything in the application environment. Bind the
four pinned runtime package versions, original requirements lock and package
plan. Verify every recorded Python source against its installed RECORD and
publish aggregate source-manifest hashes. This checks installed-file integrity
and package identity, not an independent re-download/reproduction of the wheels.

Reuse the exact [v117 measurement](native-mouth-renderer-v117-method.md) code
objects in an isolated globals dictionary with new v118 trial/cache/freeze
paths and worker entry point. Require the private source selection to equal
the preserved prefreeze selection. No image, state, seed, shift, crop,
determinism rule, counterfactual, module-count requirement or model changes.
No model result was available when choosing this runtime correction.

The worker inserts the retained overlay before importing the measurement.
The resource runner still starts the same interpreter with a clean fixed
environment. Measurement receipts retain schema v117; the new published freeze
also records execution-context schema v118 and its runtime source identities.
Tests verify exact code-object reuse, isolated globals, unchanged measurement
helpers, original argv preservation and the single worker-module substitution.

Freeze the new context before execution. All original neural limits and the
successful exact owned-descendant preflight remain required. Missing headroom
must return pending without creating an attempted model-stage directory.
Preserve any future failed/partial model attempts; do not silently retry or
switch device. All source/feature/render outputs remain private. This context
change does not qualify achieved synchronization, detector accuracy, a new
customer confidence contribution or a production deployment. No external
contact or GitHub workflow is part of this work.
