# Full video app webcam false-alarm comparison

Bind the [v200 webcam controls](video-vcd2-controls-v200-method.md) and
the underlying 12-source roster. Each recording has three planned arms,
unchanged, chroma conversion, and recompression. All 36 cases are
face-swap negatives by construction. The two ordinary edits share the
source-relative interval from three seconds up to six seconds. Compare
that interval and the separate set of native frames whose RGB actually
changed. Respect the app's 500 MiB upload limit and retain larger inputs
as unsupported. Preserve distinct source, app, guard, and evidence
unavailable outcomes across the entire plan.

Use the same [v190 full app runner](video-faceswapai-full-app-v190-method.md),
including its one-frame-batch adapter and isolated local settings.
Verify the source byte digest, exact variable picture clock and native
RGB hashes for the legacy sample and every additional retained render
anchor. Read the stored analysis in immutable SQLite mode. Check each
evidence packet's source bindings, sampled timestamps, stills, crops
and context-video picture clock against v200. Validate the observed
appearance windows even when they produce no review flag. Record
unavailable routes separately from completed measurements with no flag.

Count each published face review interval on this panel as a false flag
for face-swap review. Show overall, arm and paired-source counts.
Measure the covered native frames and their overlap with the ordinary
edit interval and changed RGB frames. Keep the app's customer direction
separate from face review flags. Replay the frozen [v194 end refinement](video-faceswapai-edge-review-v194-method.md)
on each selected appearance interval and report the old and proposed
coverage. The refinement can adjust an existing range. It provides no
independent classifier and does not change a customer direction. The
source-level comparison describes webcam controls. It does not estimate
face-swap sensitivity, causal effects across unrelated camera footage,
or lip-sync accuracy.

Run one internal APFS workspace at a time. Keep the v108 initial
3 GiB free-space requirement and ongoing 1.5 GiB reserve. Check for
room for both full source copies and temporary output before starting
a case. Review the output in a read-only process. Seal owned database,
objects, uploads, and run receipts to the mounted Seagate drive. Verify
the sealed bytes before explicitly releasing only the completed case's
workspace. Existing model references and rebuildable caches are covered
by the frozen inventory. Preserve unsuccessful attempts and their
receipts. Reports contain aggregate counts and commitments only.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_full_app_v202 preflight
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_full_app_v202 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_full_app_v202 run slot-01-unchanged
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_full_app_v202 review slot-01-unchanged
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_full_app_v202 seal slot-01-unchanged
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_full_app_v202 release slot-01-unchanged
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_full_app_v202 readout
```
