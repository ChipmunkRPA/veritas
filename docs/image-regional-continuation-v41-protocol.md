# Explicit continuation after an exit-observation failure

The original v26 fitting run stopped after phase 33. Its fitting worker wrote a
complete 32-update result and exited zero, but the monitor recorded an unresolved
`NoSuchProcess` observation around exit. The original batch remains **guarded**;
its checkpoint does not authorize continuation or held-out inference.

## Preserve the verified prefix

A new bounded audit checked the original phases 0–32: all 1,056 accepted updates,
resource receipts and journals, source-role/schedule correspondence, checkpoint
hashes, learned-parameter links, actual completion counters and preservation
declarations. The audit completed in 22.26 seconds with exit zero, no sampling
error and no resource crossing. The largest observed worker RSS sample was
207,880,192 bytes, not a guaranteed peak.

Continue from the successful phase-32 checkpoint in a separate study and private
checkpoint directory. Explicitly repeat the unaccepted phase-33 range and finish
phases 34–39: 222 accepted updates remain. All original failed files are retained.
No old result, profile or checkpoint is rewritten, copied into a fake successful
phase, or relabelled as passed. The guarded original phase-33 state is never used
as a predecessor.

The accepted model path still has exactly 1,278 updates across six epochs. Actual
compute, including the discarded 32-update attempt, totals 1,310 optimizer
updates if this continuation finishes. The discarded attempt's 64 source forwards
are accounted for separately; its recorded sources were all assigned to fit.

## Preserve the fitting method

The 397 source records, indivisible creator/source components, fit/calibration/
point/diagnostic/confirmation roles, complete source order, format/control schedule,
full input geometry, selected parameters, seeds, losses, AdamW settings and final
six-epoch selection stay fixed. No held-out source has been forwarded.

The continuation fitting body is an exact syntax-tree match to the frozen v26
body except for checkpoint/result schema identifiers and the first predecessor's
original-plan binding. It reuses the frozen source-view, geometry, differentiable
forward/loss and checkpointing implementations. Thirty-seven focused local tests
pass, including that method-equivalence check, rejection of guarded predecessors
and process-exit regression cases. Ruff passes.

## Correct the monitor prospectively

The new monitor treats a disappearing PID as terminal only after an authoritative
bounded wait returns an exit status. A worker that remains live or cannot be
observed stays a failure; a nonzero terminal exit cannot pass the phase. Resource
violations remain failures even when the worker later exits zero.

The original 900-second/6 GiB sampled worker-tree RSS limits, 1.5 GiB RAM/disk
reserves and initial 6 GiB RAM/3 GiB disk headroom remain. Sampling still cannot
guarantee a true peak. All failures are retained without automatic retries.

Publish and remotely verify the [exact continuation freeze](evaluation-panels/image-regional-continuation-v41-freeze.json)
before the first continuation forward. Use only the final completed state after
the full accepted prefix/suffix chain is audited. A later reserved-inference
supplement must bind that provenance; the original v29 auditor cannot authorize
this mixed-study chain. The frozen v27 statistical rules are not changed.

No accuracy improvement, calibrated authorship probability or serving admission
is claimed. Models and learned checkpoints remain private, serving defaults and
the protected audio calibration are unchanged, and no workflow, outreach or
cloud resource is used.
