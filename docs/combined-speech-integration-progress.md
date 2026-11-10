# Optional combined speech: presentation and export checks

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

## Customer presentation and evidence

The browser now mirrors the backend's saved-reference and source-scope checks.
Twenty-four authored cases are shared by the Python and TypeScript tests,
including malformed, skipped, tied and contradictory responses, exact window
coverage, source-clock correspondence and JavaScript-safe numeric coordinates.
The new method uses its own relative tendency, including estimates between
50% and 60%; the earlier method retains its own existing interpretation.

Audio evidence v4 and nested video evidence v8 distinguish a calibrated speech
reference estimate from raw inspection scores. A calibrated original-leaning or
tied window also retains an inspectable packet. Packet observations, waveforms,
clips and JSON coordinates use the same exact source interval. Score meanings
are validated against the saved raw run, published signal and reference hash.
Historical evidence versions remain supported without rewriting their bytes.
Only published voice checks enter those evidence manifests; comparator runs
remain private instead of supplying another finding's score or calibration.

The standard PDF describes the actual separate global calibration and limited
speech scope, rather than legacy per-context fusion, clamping or uncertainty
band claims. Its packet inference follows the same calibrated direction.
Background acoustic measurements and sound categories remain context; no new
synthetic-background capability is claimed.

Public ZIPs explicitly verify signed document integrity, not recalculation of
confidence or reproduction of inference. The private parameters are excluded.
Historical public packages without the new optional verification-scope block
remain readable; a present block cannot claim those unsupported capabilities.

## Local verification

- 472 distinct relevant backend cases have passing latest results across the
  broader run and corrective follow-ups. The broader run passed 471/472; its
  remaining audio-report scaffold was corrected to use its actual published
  run. The 98-case correction run and final 94-case report/export run passed.
- The entire frontend suite passed: 295 tests in 28 files. Type checking and
  production build passed. The bundler still notes large JavaScript chunks.
  An obsolete navigation test was aligned with the existing plain view, which
  already shows all findings; production navigation behavior was unchanged.
- Actual decoding, PCM binding, evidence generation, JSON route export,
  WeasyPrint PDF rendering and signed ZIP verification passed using an authored
  four-second tone and explicitly authored scores. AI-leaning, original-leaning
  and tied packets were checked. These are software tests, not accuracy tests.
- The final 11-page PDF was rasterized and visually checked. Its software-QA
  artifact SHA-256 is
  `dd9a4111067db1e182e8ddd315898f66bba9ee379f4f9b07fa67d0eae08d6fd3`.
  The fixture JSON route export SHA-256 is
  `1ac3844a5b4e29909e50c535e310c282343d00162e853f6bc6bcb8a7f3baa819`.
  The fixture assets and PDFs remain local and are not published as research
  results. No neural inference occurs in these authored export tests.
- Existing saved-pin routing, rollback, protected calibration, billing preflight,
  Stripe checkout and the unpaid-worker gate were covered. Stripe is mocked;
  no live payment was made. Checks found no provider names or private learned
  arrays in the tested customer exports.

## Still required before adoption

The [36-input actual extraction check](audio-combined-extraction-v98-protocol.md)
remains unqualified. Its original before-launch memory stop is preserved. A
later headroom-ready launch [stopped on a resource-monitor permission error](audio-combined-extraction-monitor-stop-v98.md),
with no saved response or result. That failed attempt was preserved without
retry. A separate owned-child permission preflight passed outside the sandbox;
it ran no model and does not qualify extraction. The frozen canonical
dependencies remain unchanged while this integration develops separately.

Actual model extraction, model leasing, a complete prepaid analysis, browser
playback alignment and rollback with the real models remain required. DOM tests
and authored export fixtures are not that end-to-end qualification. Longer
recordings, independently calibrated aggregation and synthetic-background
detection remain separate unfinished work.

This integration has not been merged or enabled in the running app. No new
accuracy, population confidence, outside human review, Actions workflow or GCP
deployment is claimed. Do not use the branch's contract tests as evidence that
the outstanding model, browser and complete paid-flow qualification passed.
