# Exact three-second audio preparation

This local stage qualifies the frozen v169b development and holdout sources
and noise parents. It materializes 48,000 samples at 16 kHz in each accepted
mono WAV. This exact three-second prefix matches the ordinary-edit controls.
The existing 47,648-sample GRID prefix is also checked for identity reuse.
[Frozen source roster](audio-short-edit-source-plan-v169b.md)

Recheck each used original archive SHA-256 once, then read the exact bounded
tar member or Parquet row. Verify declared source, speaker or donor and text
against the frozen row. Clone and expressive records also verify exact
publisher sidecar bytes and declared generation fields. Check existing
noise payloads against their frozen raw hashes.

Use the same app decoder that serves audio analysis. Verify mono signed-16-bit
production PCM and its complete mapping. A qualified prefix has 48,000
finite samples and one contiguous original source range. Retain its measured
origin. Short, silent or discontinuous prefixes are unavailable for this
controlled pilot. Compare complete original bytes, complete PCM and both
prefix PCM hashes to frozen past exclusions and every source accepted in
either role. Use the next candidate only within the same frozen connected
component when an earlier candidate cannot qualify. Keep the attempted
candidate, reason and every unfilled slot in private receipts.

Write an exact zero-origin WAV and verify production decode and sample-clock
round-trip equality. Preserve the original-to-WAV source range separately.
Output directories and receipts are exclusive and stay on the authorized
local drive. The public receipt contains only aggregate role coverage and
whole-artifact commitments. Materialization performs no detector, VAD,
analysis, billing or fitting calls. Both roles stay unscored.

The worker reuses the v108 data-only binding monitor. Its 900-second limit,
1.5 GiB sampled process-tree RSS limit, 3 GiB initial RAM and disk headroom
and 1.5 GiB running reserves apply to this preparation.

Run from the repository root.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_short_edit_materialize_v169c freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_short_edit_materialize_v169c run
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_short_edit_materialize_v169c readout
```

The guarded run requires the local process sampler and authorized archive
write access. The readout verifies the monitor receipt and the prepared WAV
commitments before printing aggregate coverage. A failure remains preserved
for diagnosis and cannot be overwritten by a second run.
