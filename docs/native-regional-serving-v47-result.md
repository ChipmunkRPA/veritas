# Native regional serving parity: complete

The reentrant app loader reproduced all **236 saved native response maps
exactly** on the previously consumed v46 confirmation inputs. Maximum absolute
difference was **0.0**. Source geometry, actual model input patches, whole-image
gates, final retained source pixels, baseline retained pixels and the actual
shared-pixel intersection also matched. The first input repeated exactly:
**237 completed neural calls**, without a fit, merge or threshold change.

This is an engineering parity result, not another fresh accuracy test. The
cohort remains 59 photos in 48 source groups, from the same editing family as
the preceding local confirmation. No source role was reassigned. The earlier
result—45/59 useful regions versus 10/59, alongside 13/59 untouched-image flags
versus 3/59—retains its original limitations and denominators.

The [protocol](native-regional-serving-v47-protocol.md) and exact implementation
bindings were manually published in `ed742df` before source inference. Remote
freeze blob: `5c4cb0b505b72f5d1a7ab0dcac9f43e2ee51459d`; SHA-256:
`ad025b7b8d9e448bac190135127f21468bc4f88db986c8884df794d3ab5c73b0`.
All 78 bound files were reverified after the replay.

All 12 monitored steps passed: one preparation, ten sequential model phases and
one seal. No guards, missing inputs, sampling failures or recorded reserve/limit
crossings occurred. The ten model phases took 442.94 seconds in total; maximum
sampled worker-tree RSS was 1,614,954,496 bytes. RSS is sampled, not a guaranteed
instantaneous peak. CPU float32, two threads, deterministic-algorithms mode
false; no GPU, other dtype or higher concurrency qualification is implied.

## Optional app integration

`VERITAS_IMAGE_REGION_MODE=candidate` enables the sensitive regional check and
retains the older recompression check for separate comparison. Explicit policy
denials still apply, and missing weights remain missing-check outcomes. The
repository and current local default remain `legacy`; no automatic rollout or
old-weight deletion occurred.

The loader reads a pinned complete 410-tensor safetensors state. It does not
download, deserialize pickle, change global imports/model factories/sockets,
merge adapters or change request-time thread settings. A per-instance lock
serializes mutable graph caches and input-binding hooks. CPU threads are set
only at owned worker startup when the optional mode is selected. Both upstream
MIT notices are included; the base weight licence and training-source rights
remain separate. Private weights, maps and source pairings were not published.

Targeted local checks cover the commercial policy/explicit denial, rollback,
PNG and minimum-size scope, corrupt/invalid/unavailable responses, concurrent
input binding, source-coordinate reconstruction and exact selected masks. Real
evidence serialization tests bind the same source crop and component-mask hashes
to report capture, JSON manifests and snapshot round trips, with public model
identities omitted. Regression checks of existing image access controls, report
rendering, public projection, Stripe checkout/preflight and unpaid-worker gates
passed. Stripe uses test doubles; no live payment was attempted.

Regional scores remain **uncalibrated editing cues with zero overall authorship
confidence weight**. Empty, skipped or failed checks do not establish an original
file. The preserved shared-mask helper and parity result do not mean a dedicated
shared-pixel UI has been completed. Full paid end-to-end app/browser verification
and that presentation remain before any local default. Ordinary non-AI spatial
edits and additional generators still need separate validation before an
AI-specific confidence contribution.

The runtime dependency extension adds only the already installed MIT helper
`einops==0.8.2`, whose official metadata declares no dependencies, to both locks.
Existing dependency pins remain unchanged. The local runtime contract and
`pip check` passed. This checkout lacks the old `audit_summary.py` required by
`make audit`; that command failed and was not represented as passed. Equivalent
direct audits covered every package in both locks with no skips. They reported
57 advisory occurrences in six existing Python packages in each graph, and the
frontend audit reported four moderate and two high findings. The added helper
had no recorded advisories. These are unresolved pre-hosting dependency work,
not a successful security/release gate. No deployment or cloud resources were
created, no people contacted, and no GitHub Actions workflows started.

Machine-readable records: [result](evaluation-panels/native-regional-serving-v47-result.json),
[execution](evaluation-panels/native-regional-serving-v47-execution.json).
