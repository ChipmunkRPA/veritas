# Combined speech engine adapter: prepared, not selected

The optional `CombinedSpeechDetector` now declares its actual four components
for model leasing and activation budgeting. It binds a complete 0–4 second
source window, calibrated estimate, series exhibit and source PCM fingerprint.
Nine local authored tests cover scope refusal, component declarations,
evidence/exhibit correspondence, skipped responses and report metadata.
No neural model is called by these adapter tests.

Reports can bind the private inference bundle through a separate descriptor
containing its whole-artifact hash, reference prior, scope and limitations.
They do not need the learned normalization arrays, coefficients, source IDs or
provider identities. The descriptor is distinct from the private bundle and
must not be represented as its contents.

This module is **not registered, enabled or used by the running app**. Manifest,
selection, fusion, exports, payment-first flow and rollback integration remain
unfinished. The [actual extraction protocol](audio-combined-extraction-v98-protocol.md)
has 36 fixed consumed fitting inputs across four source families and nine
conditions. Its [admission controller stopped before launch](evaluation-panels/audio-combined-extraction-admission-stop-v98.json)
after 900.65 seconds without two sufficient headroom observations. At that stop,
no model-stage folder or response output existed.
The controller allowed 128 MiB for the separate monitor above the unchanged
6-GiB neural startup minimum. This is a resource wait, not a candidate failure.
The later [headroom-ready attempt stopped on a monitor permission error](audio-combined-extraction-monitor-stop-v98.md).
Its worker was launched and terminated without a response record or result.
That failed qualification is preserved without retry. The successful separate
permission preflight ran no model. Keep the frozen extraction dependencies and
serving default unchanged; actual extraction remains unqualified.

Further app integration is being developed on `codex/combined-audio-serving`
in a separate checkout. Its source-timestamp, calibration-selection and report
changes have not been merged into the running app. Outside the frozen test,
that adapter now requires a source PCM hash and one complete continuous source
time range, refusing gaps and decoder-limited excerpts. Integration checks are
software contracts and do not replace actual model extraction or paid-flow
qualification.

The existing current detector and protected calibration stay intact. Raw media,
individual source/feature fingerprints and learned arrays remain private. All
checks and publication are local/manual; no Actions, outreach or GCP work.
