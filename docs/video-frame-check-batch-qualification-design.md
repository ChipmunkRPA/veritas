# Qualify the skipped frame check at a bounded video batch size

In the [first serving batch](grid-serving-baseline-v154-partial-coverage.md),
the frame authenticity ensemble was skipped on all eleven cases. The model
manager allowed 4,608 MiB; its activation plan alone requested 6,400 MiB.
Those are accounting figures. The full worker reached 5.18 GiB of sampled
descendant RSS without this route. The memory of a restored route needs a new
measurement. Do not raise either budget for this test.

Test a separate video-only route with one face per forward. Bind the same
limit in its activation declaration and actual prediction call. Keep both
verified commercial checkpoints, the 380 px crop and preprocessing, largest
face selection, frame order, and mean of the two sigmoid outputs. The current
estimate would reserve 800 MiB for activation plus at least 510 MiB for the
resident weights; actual measurements may differ. Keep the full registry,
licensing policy and selected routes. The still-image adapter and its
aggregation stay unchanged. See the [video adapter](../backend/app/detectors/video/adapters/frames.py)
and [shared classifier](../backend/app/detectors/image/dfdc.py).

Qualification needs a valid same-device reference. The current serving budget
cannot admit the batch-eight route. If a reference cannot be measured under
the frozen host limits, report parity as unresolved. Compare crop bytes,
frame times, both members' responses, ensemble and aggregate scores, and
evidence selections, including short final batches and missing faces. Test
cold and warm full serving at the existing frame limit, with measured
forward sizes, device, dtype, sampled RSS and GPU allocations. A new route
needs its own identity: the released compatibility check matches weight
identities and does not establish execution-path parity.

On matched originals, known swaps and ordinary controls, compare the current
configuration with the bounded route. Keep skipped and unavailable results in
the denominator. Report missed edits, false alarms, localization and
evidence delivery. GRID is suitable for an engineering coverage test; its
mouth redraws do not establish face-swap accuracy. The
[historical ablation](v2/bench/admissions/vb-1.1/video/ablation-vid.frames.dfdc_b7.md)
did not pass incremental admission. Restoring model coverage cannot by
itself justify changing production defaults or the overall estimate.
