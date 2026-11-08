# Regional image learning: initial execution progress

The first two v26 fitting phases completed successfully: **64 of 1,278 planned
paired optimizer updates**, with 128 source forwards and 128 backwards. This is
actual supervised adaptation of the backbone's local/global LoRA paths and
region decoder. Detection accuracy has not yet been assessed.

The [prospective protocol](image-regional-supervised-v26-protocol.md) and
[preparation freeze](evaluation-panels/image-regional-supervised-v26-freeze.json)
were published at `c9be0dc2a153e0a3ff991dcd7a4f17492a7810ff` and remotely verified
before the first project-source forward. The
[aggregate progress receipt](evaluation-panels/image-regional-supervised-v26-progress.json)
records the completed phases, private-checkpoint hashes and local readback
reviews. No source identities, learned tensors or optimizer moments are public.

## Verified so far

- All 246 selected tensors, comprising 6,009,089 parameter elements, changed in
  each completed phase. Both backbone adaptation paths and the decoder changed.
- The dense base, fixed positional frequencies and protected voice calibration
  remained unchanged. No serving or customer-confidence changes were made.
- Both saved learned states reloaded with exactly matching tensor hashes.
  All 246 AdamW moment states were finite, with steps 32 and 64 respectively.
  Phase 1 began from phase 0's exact learned tensors and checkpoint hash.
- All 64 consumed photos belong to fitting groups. No calibration, point,
  diagnostic or confirmation source was inferred or fitted.
- Phase durations were 297.45 and 298.38 seconds; maximum observed worker-tree
  RSS samples were 1,736,065,024 and 1,761,099,776 bytes. Neither phase crossed a
  resource limit or had a sampling error. These are sampled lower bounds,
  not guaranteed peak memory figures.
- 39 relevant local tests and lint passed before execution. GitHub Actions
  remained disabled; the preparation commit had zero workflow runs.

## Remaining work

The remaining explicitly numbered phases 2–39 are being executed sequentially,
without retries or skipping a failed phase. Each worker uses the frozen
per-phase resource monitor and exact predecessor checks. Private checkpoints
stay on the owner's connected Seagate drive; the continuation program's hash
is recorded in the aggregate receipt. A future status must inspect actual
processes and completed profiles rather than assume this sequence is running.

The first two phases cover native images and original-image negative controls
only. JPEG75 and matched tone/blur negatives occur later in the prescribed
six-epoch schedule; they have not yet been demonstrated by these two phases.

Use only the final six-epoch checkpoint. Before reserved-source inference,
freeze the readout, calibration, false-alarm comparison, uncertainty method and
confirmation decision separately. The 184 reserved photos cover 32 calibration,
24 point, 48 diagnostic and 48 confirmation groups. There are 59 confirmation
photos, not 59 independent creator groups.

No improvement over the current detector is established by training loss or
successful updates. Unseen generators, unresolved source-owner overlap and
external review remain outstanding. No outside person was contacted.
