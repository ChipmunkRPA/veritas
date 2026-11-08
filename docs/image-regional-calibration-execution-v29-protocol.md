# Final training audit and calibration admission, v29

This completes the preparation path connecting the running v26 training study
to the frozen v28 calibration producer and v27 statistical readout. It does
not execute an audit of unfinished training or admit neural inference yet.

## Before calibration

Wait for the actual training sequence and final worker to terminate successfully.
Do not infer completion from a journal or checkpoint filename alone. Run the
model-free preparation worker through its dedicated resource monitor. It refuses
immediately if phase 39 lacks both a completion receipt and terminal profile,
then verifies the full chain rather than only the last checkpoint.

The inherited v28 chain auditor verifies all 40 exact resource-profile identities,
their successful terminal outcomes and resource-journal hashes; all 1,278
prescribed updates and their source roles; unchanged parameter/checkpoint links;
each checkpoint's retained bytes; unchanged dense/frequency state and native
readback declarations; and zero reserved-source fitting/inference counters.
Actual detection accuracy is outside this audit's scope.

The preparation worker separately verifies the published v26/v27/v28/v29 source
commitments, private training plan, pinned image-only research checkpoint and
both current baseline checkpoints/backbone. It records Python/architecture,
Torch, Torchvision, timm, NumPy, Pillow and installed OpenCV distribution versions.
Missing required distributions or changed assets fail preparation; no package
upgrade, download, weight substitution or new training is performed.

The data audit is a separate sequential phase, limited to 900 seconds and
6 GiB sampled worker-tree RSS, with 1.5 GiB RAM/internal/external-disk reserves
and the same initial headroom rules as the earlier studies. Retain all failed
profiles. Do not overlap this audit with training or another heavy model phase.

## Prospective execution supplement

After successful checks, create a new private trial containing:

- `training-chain.json`: retained phase hashes and final learned-state identity;
- `private-paths.json`: the private study, raw-map destination and preparation
  profile location, kept out of the public supplement;
- `calibration-tasks.json`: all 156 fixed source/kind tasks per arm/format;
- `freeze.json`: aggregate counts and hashes, exact implementation/asset/runtime
  identities, the final learned-checkpoint hash and zero reserved forwards.

Use an empty private map destination. Existing trial outputs cannot be reused
or overwritten. A preparation output alone is not successful execution: wait
for and inspect its terminal resource profile. Publish the reviewed supplement
without the private files, verify the exact remote bytes, and only then write
the local execution-start receipt binding that publication and successful audit
profile. No GitHub workflow is involved.

## Calibration command boundary

Use the v29 score entry point to verify publication, code, private paths, runtime
versions, the exact terminal preparation-profile identity and its observation
journal before invoking the frozen v28 inference monitor. The underlying
producer separately verifies the final learned checkpoint, training-chain and
role commitments before loading or scoring a reserved input.

Run each explicitly named arm/format/phase sequentially. There are 28 phases:
seven each for candidate/native, baseline/native, candidate/JPEG75 and
baseline/JPEG75. Preserve all 624 unique model-input calls and the four
first-input repeat checks. Model/setup/data failures retain unavailable or
failed outcomes; they do not silently remove sources or yield an original-file
verdict. The readout requires complete data and follows the fixed v27 policy
grid and source-group rules.

No score command is authorized by this preparation record yet: training is
incomplete. The final checkpoint has not been declared or tested. Full chain
auditing, publication of the actual execution supplement, calibration model
calls and map-policy selection still remain.

## Local checks and interpretation

Fourteen fixture-based tests and lint passed. The checks cover early refusal
while training is active, occupied destination protection, complete task/count
assembly without public path disclosure, exact runtime/implementation/publication
binding, failed or unrelated audit rejection, resource-journal integrity and
refusing to start an audit without memory headroom. Test doubles exercise
assembly; they do not establish completed real training or neural performance.

Preserve this version's source/tests as recorded in the preparation artifact.
An implementation correction requires an explicit successor. Serving admission,
probability calibration, broader fresh generators, external review and full
app evidence/export verification remain separate unfinished requirements.
