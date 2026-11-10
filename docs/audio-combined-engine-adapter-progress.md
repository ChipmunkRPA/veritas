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
conditions. Its admission controller is waiting for memory; no extraction
result is claimed here. Keep the frozen extraction dependencies unchanged
while that test is pending.

The existing current detector and protected calibration stay intact. Raw media,
individual source/feature fingerprints and learned arrays remain private. All
checks and publication are local/manual; no Actions, outreach or GCP work.
