# Native video appearance: six batches bound to actual inputs

The first six fixed batches in the [native method](video-appearance-native-v108-method.md)
completed locally. **108 clips, 3,564 scheduled frame inputs and 324 native PNGs**
matched their file, pixel and rational-time commitments. This completes part of
input qualification; detector scoring and an accuracy comparison have not run.

The [aggregate snapshot](evaluation-panels/video-appearance-binding-progress-v108.json)
was recorded at **08:24 UTC on 10 November 2026**. It is a point-in-time receipt,
not a live progress file. Keep it and the individual private receipts unchanged.

## What passed

| Item | Verified at this snapshot |
| --- | ---: |
| Selection groups | 18 of 23 |
| Constructed clips | 108 of 138 |
| Publisher-swap conditions | 36 |
| Original/ordinary-edit controls | 72 |
| Planned windows bound | 540 |
| Scheduled native frame inputs | 3,564 |
| Construction PNGs bound | 324 |
| Completed binding batches | 6 of 8 |
| New neural calls | 0 |

Every group retains all six conditions. The binding stages rehash actual MP4
files, probe their complete frame tables and display durations, decode the
scheduled RGB frames and verify three before/during/after construction images.
The images refer to the same scheduled interval for controls and swaps; they
are not suspicious regions discovered by a detector. Only the explicitly
listed source pictures are sampled by the planned appearance check.

The original/ordinary-edit controls are reference labels from publisher footage
and our fixed construction recipes. They are not an independent forensic
certification of the publisher's footage. Changed swap-source frames and
publisher provenance do not establish exact AI-pixel or face-identity truth.
Existing source checks found no exact historical RGB-sequence matches; unknown
shared people, re-encodes and pretraining overlap remain possible.

## Execution and software checks

All six processes exited zero with no guard or sampling failure. Binding time
totaled **642.62 seconds**. Maximum sampled descendant RSS was **534,937,600
bytes**, and minimum observed available RAM was **5,367,021,568 bytes**. These
are sampled observations, not guaranteed peaks. Verified short-lived descendant
exits remain recorded separately from sampling errors. Data-stage limits stayed
at 900 seconds per batch, 1.5-GiB sampled RSS, 3-GiB initial RAM/disk and 1.5-GiB
running reserves.

The owned-process preflight successfully enumerated and sampled its live
standard-library child/grandchild and verified exit. Its internal live sample
was 27,836,416 bytes. This resolves process-access readiness for this new stage;
it does not rerun or change the failed historical audio attempt.

The implementation passed **63 local tests** across the new native helper,
support plan, retained calibration and command-review arithmetic. Thirty-one
tests belong to the new native helper/runtime. They cover altered clocks,
geometry and labels, missing context, strict neural resource admission,
preservation of prior attempts, and readout parity with the retained appearance
branch plus a separate scalar-sum implementation. Ruff and diff checks passed.
These tests load no pretrained network or source media.

A subsequent metadata review checked all six complete receipt/profile chains,
their worker/result/resource hashes, exact prepared case order, scheduled frame
sets and rational clocks. This is local recorded-data verification using the
existing codec implementation, not an outside human or independent-codec review.

## Still unfinished

Two batches and 30 selection clips remain to bind. The constructor had 73 of
90 source groups command-reviewed at the snapshot; construction continues under
its original fixed plan. All 24 confirmation groups remain excluded from this
appearance transfer comparison.

The authored neural contract and all source feature-extraction batches remain
unattempted. A read-only admission check reported `initial_ram_headroom`; it
created no attempted model stage. Neural admission still requires 6 GiB of
available RAM, followed by the unchanged neural runtime/RSS/reserve limits.
No skipped check has become an original-file verdict.

After actual native extraction, compare both error directions, unavailable
cases and sparse time/frame support for the fixed raw cue and sealed file map.
The map uses the same underlying signal, so a gain in flags is an operating
point/calibration tradeoff, not a new neural capability. Lip interaction,
background-audio methods, full production tracking/payment/export validation
and customer-confidence qualification remain separate work.

The freeze was pushed before execution in `333260c`. Source media, individual
fingerprints, pairings, face features and fitted maps stay private. Serving
defaults and rollback are unchanged. GitHub Actions was verified disabled;
no workflow, outreach, customer analysis, real payment or GCP operation ran.
