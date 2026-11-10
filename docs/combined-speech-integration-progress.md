# Optional combined speech integration: backend phase

Development branch: `codex/combined-audio-serving`. This branch is separate from
the checkout serving the local app. Its new option defaults to false; no running
default or protected calibration has changed.

## Implemented and checked locally

- Register the pinned complementary encoder and private inference bundle with
  their separate notices and model-memory declarations. Keep commercial-policy
  checks over all dependencies, and keep the current/legacy detectors installed.
- Select the new calibration only for its explicit option and a four-second
  decoded input, or the matching saved calibration pin. Other durations retain
  the existing path and its existing scope rules; no window averaging creates
  a whole-file probability.
- Require exact source PCM correspondence, a complete continuous source clock
  and a decoder limit beyond the checked window. Refuse gaps, incomplete maps
  and decoder-limited excerpts. Retain original source PTS in the evidence.
- Keep the selected calculation's score, explanation and finding together.
  Comparator runs stay in the detection record and are not double-counted in
  the audio conclusion. Unsupported/missing combined evidence supplies no
  calibrated confidence or original verdict.
- Recognize the new saved reference in backend customer/finding presentation
  and disclosures, using its own frozen relative tendency. Do not borrow the
  old stability guard or obsolete error-rate wording.
- Store a report descriptor binding the private calibration by hash, without
  distributing its learned arrays or provider identities. Preserve source and
  pretrained-weight notices separately from the application's MIT code.

Targeted backend tests passed for these contracts, saved-pin routing, existing
customer/finding/export behavior, the historical calibration fixture, billing
preflight, Stripe checkout and the unpaid-worker gate. These are software and
recorded-data checks, not actual new model extraction or a full paid analysis
through the optional path. No live charge was made.

## Still required before adoption

The [36-input actual extraction check](audio-combined-extraction-v98-protocol.md)
has not started: its before-launch memory admission stop is preserved on main.
All model inputs remain never attempted. The frozen canonical dependencies
remain unchanged while this integration develops separately.

Finish the browser's mirrored customer/finding reference handling, including
scores between 50% and 60%, without changing the old reference's interpretation.
Test the actual new PDF, JSON and verification-package exports. A hash descriptor
is a calibration reference, not independent reproduction of its private arrays;
package verification must state that limit. Then test checkout-before-inference,
model leasing, a complete analysis, source playback alignment and rollback with
the actual models. Longer recordings and independently calibrated aggregation
remain separate unfinished work.

This backend phase has not been merged or enabled in the running app. No new
accuracy, population confidence, outside human review, Actions workflow or GCP
deployment is claimed. Do not use the branch's contract tests as evidence that
the outstanding model, browser and complete paid-flow qualification passed.
