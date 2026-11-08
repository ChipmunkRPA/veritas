# Regional learning: reference data verified; full corrected probe still pending

**All397 source-bound image/edit/mask pairs are acquired and verified.** The
separate successor review reproduces every changed-pixel truth mask, all794
ordinary-control RGB commitments and all794 source/edit dual-decoder checks.
The source graph has320 connected new-source groups. No project-image neural
inference, supervised fit, new accuracy gain, default replacement or customer
confidence contribution is established by this result.

The [aggregate result](evaluation-panels/image-region-data-and-training-probes-v23.json)
contains actual counts, resource profiles, source/program commitments and
preserved failures. Photos, pairings, private sketches/indexes, gradients and
weights are excluded. GitHub Actions remains disabled; checks ran locally.

## Completed reference data and source graph

The [v21 plan](image-positive-pairs-v21-protocol.md) retains all317 publisher
training,61 validation and19 testing photos. All397 pairs pass source-size,
nonempty edit and changed-pixel containment checks. The two ordinary controls
use the same source and alpha without a generator. Neither a source/control
label nor a changed-pixel mask independently authenticates camera or authorship.

Acquisition has18 completed phase records, including one copied successful seed
and17 newly executed phases. The whole chain contains2385 stored HTTP receipts
for19,996,798,159 transferred compressed bytes. Repeated range commitments agree.
The review binds794 selected members across six archives. Imported indexes still
disable gzip CRC checks; the audit makes no signed whole-archive checksum claim.
Only selected assets/private indexes were retained, rather than whole archives.

The original separate reviewer failed on an incorrect nonempty-index assumption.
Its frozen code and failed receipt stay unchanged. The
[explicit format successor](image-positive-pairs-v22-index-correction.md),
published as `02882a25d2977afa4133a20be2cde46239bf46fe` and remotely verified
before execution, passes the complete chain and all-pixel replay. That takes
36.047 seconds with287,162,368 bytes sampled RSS and no sampling error/guard.
This is a second local implementation, not outside human review.

The [frozen source graph](image-regional-learning-v22-preparation.md) also reads
all240 previously consumed originals. All637 originals form560 components:
320 containing new sources and240 prior-only components. New-source links are
attributed-profile links; the fixed exact-RGB/thumbnail checks detect no link
to the old sources. This does not prove absence of crops, bursts, aliases,
shared old-source photographers or external checkpoint-pretraining overlap.

There are77 new-source components containing a publisher validation/testing
source and no detected prior-source link. They are candidates for prospective
role review, not automatically fresh confirmation. Current photographer checks
for the240 older sources remain incomplete. No fitting/calibration/point/
diagnostic/confirmation role is assigned here. The earlier48-source confirmation
minimum is not reduced. The graph phase takes7.097 seconds and90,570,752 bytes
sampled RSS, with no guard or sampling error.

## Trainability probe: failure found and corrected, full successor not passed

The full1024 CPUfloat32 authored probe uses the complete strictly restored
RelayFormer graph; source is MIT and the official image-only weight card declares
Apache-2.0. It trains the actually used small adaptation layers and decoder in
memory while holding the dense base fixed. The geometric target is deliberately
authored, not an example of real or AI-generated content. No project photo is
passed through this neural experiment and no new weight file is saved.

The original v22 selects249 tensors. Its first update completes with finite,
nonzero gradients and loss1.3190274238586426. It fails the second repeat's
gradient check after four authored forwards, two backward passes and one update.
An initial conversational update incorrectly said no update ran; inspection of
the complete log establishes that the failure occurred on the repeat. The
original run remains failed, with its exact code/log/resource receipt retained.

An explicit first-pass diagnostic confirms native inference parity and finite
gradients for all249 initially selected tensors. Inspection and an actual small
library regression reproduce the cause: three decoder positional-frequency
parameters are declared fixed by the rotary helper, which returns detached
cached frequencies after the first call. Selecting them for training was wrong.
The v23 successor freezes those three values while keeping all actual backbone
adaptation/decoder paths selected. Eight additional local regression tests pass.

The full v23 successor **does not pass yet**. Its resource monitor terminates it
after10.817 seconds, with guard `memory_disk_or_time`, no sampling error and
2,839,658,496 bytes sampled RSS. Individual limit values were not recorded at
the stop, so its exact instantaneous trigger is not independently established.
Immediately subsequent checks find only1,344,114,688 free disk bytes, below the
1,610,612,736-byte reserve. The completion receipt is absent and its incremental
gradient journal is empty; complete network/update counts are not claimed for
this interrupted attempt. It is preserved and has not been automatically retried.

The original probe and diagnostic peak sampled RSS are3,792,912,384 and
3,561,439,232 bytes respectively. These are sampled lower bounds, not guaranteed
instantaneous peaks. The original acquisition sampling failure also stays failed;
none of the later results clears it.

## Remaining work

Restore sufficient local disk capacity before another full-model attempt; the
owner has been asked to free space. Verify the fixed-frequency probe in a new
explicitly recorded phase, then freeze connected-source roles, full training
recipes, its own calibration, comparisons and confirmation before project
learning. Local quality, unseen-generator transfer and the complete app path
remain required before adoption. The existing serving checks, voice calibration,
rollback assets and prepaid analysis flow remain unchanged. Outside human
reviewers remain0; GCP is deferred.

[Recent regional-detection research](regional-detection-research-2026-11-08.md)
supports testing unchanged backgrounds, real-source/AI-source distinctions,
sequential edits and compression. Non-commercial benchmark data is not imported
into the commercial project. Research-method reading, successful data review and
small gradient tests are not substitutes for a measured detection improvement.
