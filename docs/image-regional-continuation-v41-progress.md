# Repeated fitting batch verified; final suffix running

The [explicit continuation](image-regional-continuation-v41-protocol.md) has
completed its first repeated batch successfully. Its monitor recorded terminal
exit zero, no sampling error and no resource crossing. There are now **1,088
accepted training updates**, with the remaining 190 queued once, sequentially.
This is a progress snapshot, not a completed training or accuracy result.

A separate bounded, read-only review loaded the actual saved checkpoints with
restricted tensor deserialization. All 246 learned tensors, the complete optimizer
state and parameter groups, and the full update journal exactly match the
completed numerical calculations of the discarded original attempt. The initial
parameter chain also exactly matches the last successful phase-32 checkpoint.
This demonstrates faithful resumption without resetting training mathematics.

The original guarded checkpoint was comparison evidence only. It was never used
as the continuation predecessor, and its failed resource profile remains failed.
The new batch is independently computed from the successful checkpoint and has
its own successful monitor receipt and private saved state.

The fitting phase took 302.13 seconds with largest sampled worker-tree RSS
1,754,923,008 bytes. The read-only comparison took 1.34 seconds with sampled RSS
301,580,288 bytes. Both completed with exit zero and no monitor/resource error;
these samples are lower bounds, not guaranteed peaks. No held-out source was
forwarded and no neural call occurred in the read-only comparison.

The [aggregate receipt](evaluation-panels/image-regional-continuation-v41-progress.json)
binds the saved state and review by hash. A complete accepted prefix/suffix audit
and a prospectively published reserved-inference supplement are still required
before calibration. Serving defaults, scoring rules and the protected audio
calibration remain unchanged. Learned states and source pairings stay private;
no workflow, outreach or cloud resource is used.
