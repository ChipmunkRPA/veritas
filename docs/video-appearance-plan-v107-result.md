# Native appearance transfer: source support prepared

The [frozen method](video-appearance-plan-v107-method.md) completed its
source-only preparation locally. It schedules a fixed comparison of the
existing raw appearance cue with the sealed v106 appearance-only calibration.
No detector was loaded, no new accuracy was measured and no serving default
changed.

## Prepared support

| Item | Count |
| --- | ---: |
| Selection source groups | 23 |
| Clips | 138 |
| Publisher-swap conditions | 46 |
| Genuine/ordinary-edit controls | 92 |
| Planned appearance windows | 690 |
| Distinct planned model-frame inputs, summed per clip | 4,554 |
| Planned neural batches | 8 |
| New neural calls | 0 |

Every case retains five fixed windows, with nine middle samples and twelve
context samples per window. All 46 publisher-swap cases have at least one
planned middle sample with a changed swap-source frame. That does not mean
every manipulated frame was sampled or that a detector found any edit. Changed
source pictures and publisher provenance are distinct from exact AI-pixel or
face-identity truth. Context-only changes do not become positive middle truth.

The eight batches contain 3, 3, 3, 3, 3, 3, 3 and 2 source groups. All six
conditions are retained for every group. The constructor's other 21 fitting,
22 calibration and 24 confirmation groups were not used by this preparation.
The known source roles do not establish independence from unknown model
pretraining data. These are silent controlled excerpts, not a lip-sync test.

## Local verification

All 15 authored tests passed, including rational/NTSC picture support,
half-second tie handling, context-only edits, zero-change provenance, ordinary
edit labels, unavailable short inputs and invalid geometry. Ruff and diff
checks passed. A separate standard-library sanity check verified all 138
distinct group/condition pairs, six complete conditions per group, five
windows per case, sample/context separation and the private plan commitment.
This is local software verification, not an outside review or accuracy test.

The data-only preparation exited zero in 0.479 seconds. Its between-group
resource observations recorded 134,512,640 bytes maximum self RSS and
5,739,003,904 bytes minimum available RAM. These are sampled observations,
not guaranteed peaks. The existing 900-second/1.5-GiB data-stage budget,
3-GiB initial RAM/disk admission and 1.5-GiB running reserves were unchanged.

- Private plan SHA-256:
  `1a238e8afbe7e68d4920de765c4c6f5d8eed5c7b7540ee28beb0fee844354848`
- Preparation freeze SHA-256:
  `c39d4e30de71f29db7dac373d048f65dc3f5735e24cb1fb15a2641cb57b84ae7`
- Resource journal SHA-256:
  `c49ed55953110b3252c93ce12f1198d93bf4a16e9ae1d32e6cf674ff9efa74c1`
- Public [aggregate receipt](evaluation-panels/video-appearance-plan-result-v107.json)
  and [freeze](evaluation-panels/video-appearance-plan-freeze-v107.json).

## Still required

The scheduled clocks are explicitly expected from source recipes, not new
observations of all constructed files. Bind the complete constructor and
command-review receipts, actual encoded bytes, decoded RGB and rational clocks
before extraction. Qualify native feature/readout behavior, then compare
timestamps, false alarms, missed edits, unsupported cases and cost on these
selection inputs. The v106 maps and research points remain fixed; no new fit
is part of this transfer comparison. Full production tracking/fallback,
lip/audio inference and customer exports are separate qualifications.

The two cached commercial weight records were hash-checked only. Existing
neural admission still requires 6 GiB available RAM, with unchanged 900-second,
6-GiB sampled descendant RSS and running reserve limits. Private source IDs,
pairings, per-frame hashes and learned parameters were not published. No
customer analysis, payment, outreach, Actions or GCP operation occurred.
