# Fresh native mouth observations for the GRID pilot

The observer supplies measured geometry for the separate trajectory readout.
Use the completed v143 pilot plan, actual per-clip v149 construction receipts,
and the isolated v132 Face Landmarker runtime and v144 CPU `IMAGE` settings.
Verify their frozen sources, installed wheel manifest, permitted task assets and
reviewed bundled-model cards before the worker observes any pictures.
[Trajectory design](grid-achieved-motion-validation-design.md)

Each pilot has 285 requests. Measure the remuxed original at frames 15–58,
each of the ten numbered variants at frames 25–48, and an exact independent
repeat of original frame 27. Bind every request to the original rational PTS
and constructed PTS and RGB commitments. For ready inputs, hash the owned case
file, probe the complete native frame table, and decode the selected frames.
Check shape, index, PTS and plain RGB hash before passing a fresh RGB copy to
the observer. Condition labels stay outside the observer callback. Missing
construction retains its request slots and reason without a task input.

Require one complete finite 478-point face mesh. Keep only the thirteen used
native 2D landmarks and the four eye-normalized lip shape values. Save eye
distance, roll and nose projections in canonical hexadecimal float form. The
descriptor uses the three inner-lip gaps and projected width defined in the
design. Retain missing, ambiguous, nonfinite and unsupported geometry with
explicit reasons. Every safe scheduled picture gets its own task call.

Check the original repeat, the unchanged-RGB audio delay, and the genuine
frame-copy control. The latter uses the actual 320 ms earlier reference inside
the edit and the same-time reference on the two flanks. Independently observed
geometry or missing status must agree for the exact same RGB inputs. Retain
failed control observations with their partial stage evidence. For variant
pictures, compare 7-by-7 patches centered on the original eye vertices with the
same-time original. When they match, record the RMS drift of the landmark
estimates on those identical pixels. Keep patch and geometry coverage explicit.

Run one exclusive `clip-OO/authored` worker per pilot through the unchanged
v108 neural monitor. It uses 900 seconds, 6 GiB sampled descendant RSS and 6
GiB initial RAM, 3 GiB initial project and archive disk, and 1.5 GiB running
reserves. The worker checks for a live resource journal, an unfinished profile
and both output streams attached to the monitor log before creating its output.
Flush task starts, completions and observation records as they occur.
Replay ordered task events and canonical descriptor arithmetic at completion.
The stage receipt counts Face Landmarker API inputs. Preserve partial attempts
and separate subsequent guarded stages.

`scripts.grid_mouth_observer_v157.load_completed(ordinal)` checks the actual
worker result, resource samples, bound input, observation and task journals,
control checks and aggregate counts. It returns the picture shape, source slot,
component index, rational `source_clock`, private case-number mapping and all
285 `observations`. Each observation contains native picture and clock bindings,
availability, and `descriptor_hex` and `eye_distance_px_hex` when geometry is
available. `repeat=true` marks the repeat request for the downstream readout.
The loader uses metadata only. The observer journal and checks stay in the
per-clip Seagate cache. Publish aggregate counts and whole-artifact hashes.
Individual landmark coordinates, clocks, case mappings and RGB hashes stay private.
