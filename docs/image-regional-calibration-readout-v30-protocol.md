# Calibration map replay and outcome reporting, v30

This preparation implements the fixed v27 statistical readout for the v28
producer's private maps. Training is still active. No reserved map exists yet,
no real calibration policy is selected, and no detection-quality result follows
from these implementation checks.

## Admission and matching

The replay first requires v29's exact final-training/publication/runtime/audit
admission and a separately verified publication receipt for this readout,
bound to the same final execution-freeze hash. It then requires all 28 producer
phases to be complete: both arms and both formats, all seven phases each.
Every terminal resource profile, phase identity, source/kind row, forward and
repeat count, output journal and execution-freeze commitment must agree.
A failed or missing phase cannot be dropped to make calibration succeed.

For each matching pair, reproduce the source-bound view through the frozen
source/control constructor and one-time native/JPEG75 encoder. Verify encoded
input bytes, decoded RGB, dimensions and the precompression truth identity for
both arms. Negative controls use a zero AI truth field. A JPEG's changed pixels
are not new AI labels.

Accept only exact retained float32 maps inside the declared private raw-map
destination, with matching file hashes, recorded shapes and finite 0–1 values.
Candidate maps must cover the original source dimensions. Baseline maps must
have their exact current 384/512 dimensions. Reconstruct the baseline's image
eligibility and compare it to the producer's record; do not change its gate.

## Map and policy replay

Use the existing score-map helper, screening grid, 8-bit rounding, source
reliability exclusions, connected components, minimum areas, display caps and
nearest source projection. Use one OpenCV thread for readout phases. This
measures one retained map, not global evidence-packet admission or the full app.

Compute all seven pixel settings once per candidate map and apply the four
fixed area gates, producing the exact 28 v27 policy measurements. A closed gate
has an explicit empty selected mask while retaining ungated diagnostics; it
does not imply that the source file is original. Retain true positive, false
positive and false negative pixel counts against the entire source truth.

Run 14 explicit model-free replay phases, seven per format, and a separate
model-free policy-selection phase. Their dedicated monitor preserves the
900-second/6-GiB sampled-tree-RSS limits, initial headroom and RAM/internal/
external disk reserves. Existing outputs and failed profiles are never
overwritten, skipped or implicitly retried. No model is loaded by the replay.

Selection requires all 156 views per format and successful terminal readout
profiles. It delegates the policy grid, source-group weighting, false-alarm
constraints and conservative tie breaks to frozen v27 code. If no policy is
eligible, leave the selection and candidate outcome summary empty. Do not
report a favorable zero-detection or original-file conclusion instead.

## Outcome and coverage reports

For each selected format/arm, report edited-photo counts, photos without any
retained region, photos below the fixed useful-localization standard and flagged
original/tone/blur controls. These per-photo descriptions complement the primary
component-weighted summaries; correlated photos are not extra independent groups.
Report producer statuses with all 156 planned views per arm/format in the
denominator. Failed, unavailable, skipped, inconclusive, missing and duplicate/
malformed records stay visible. An optional metadata-only coverage report does
not grant calibration admission; an `ok` row alone cannot substitute for a
successful terminal resource profile or verified map.

The readout makes no neural calls and performs no probability calibration.
Localizer responses, internal policy settings and these benchmark rates are not
whole-file authenticity probabilities. Point/diagnostic/confirmation scoring,
broader fresh generators, full app evidence/export verification and any default
change remain separate unfinished work.

## Local verification

Seventeen local tests and lint passed. Tests use authored response fields and
metadata fixtures, not real model outputs. Native/JPEG tests exercise the actual
screening-map helper, preserve truth coordinates and prohibit neural-model
loading. Other checks cover flagged non-generative controls, exact map identity,
invalid dtype/shape/value/path/bytes, mismatched inputs/truth/gates, skipped checks,
complete unknown denominators, failed selection, resource preflight and explicit
empty policy selection. No detection accuracy has been established.

The preparation record commits this source, resource monitor, tests and document.
Preserve this version; an implementation correction requires an explicit
successor. Publish and verify its actual preparation bytes before a real replay.
