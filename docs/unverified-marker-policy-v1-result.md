# Copyable origin markers: policy correction and local verification

Reviewed 7 November 2026 (local date). This is a policy-correctness test, **not a
measurement of deepfake-detection accuracy**. The provider inventory still covers
[37 families and 51 scoped signal records](provider-provenance-coverage.md);
research coverage does not mean every production watermark has a working decoder.

## Reproduced failure

The baseline was `e87dd70a639a4bcfa79ee29770b3a35b9b5fb6c0`. Four authored PNG
controls used deterministic gradients and a sine wave, with no AI generation:
a plain image, the same pixels with an unsigned AIGC label, the same pixels with
generation-style parameters, and a copy carrying a public, unkeyed watermark.
The native watermark decoder matched 44 of 48 bits and reported presence.

The pre-model gate correctly continued processing in every case. However, a
later fusion rule let two copyable markers force an `ai_supported` decision
and an **uncalibrated** 0.85 inspection score when no learned check had run.
This was not an 85% confidence estimate. The customer interpretation treated
that marker-led answer as a provenance record, bypassing the normal no-result
presentation.

| Control | Baseline decision / inspection score | Corrected decision / score |
|---|---|---|
| Plain authored pixels | Not assessable / none | Not assessable / none |
| Unsigned structured AIGC label | Not assessable / none | Not assessable / none |
| Generation-style parameters | AI supported / 0.85 | Not assessable / none |
| Embedded public watermark | AI supported / 0.85 | Not assessable / none |

The before/after runs used the exact same input checksums in separate isolated
application databases. All eight analyses completed; each attempted further
checks. Model leases deliberately raised an error, so there were **zero learned
forwards, zero provider-verification requests and no customer database access**.
The test establishes behavior under missing content evidence, not real-photo
false-alarm rates, sensitivity or performance across generators.

## Corrected policy

- Unsigned labels, generation parameters, public unkeyed messages and untrusted
  credential declarations remain recorded observations. They do not force a
  result, add AI-origin confidence or skip applicable content checks.
- Trusted, valid, current-file-bound credentials explicitly declaring AI can
  still stop statistical work before model acquisition. Missing, unsupported,
  expired, untrusted and invalid credentials do not establish originality.
- A combined calibration containing a retired marker input must be revalidated
  before publishing confidence. Merely substituting its fitted missing-input
  response would not establish that the changed feature set remains calibrated.
  The packaged image calculation has no such marker inputs; adding a marker
  leaves its calculation unchanged in the regression checks.
- Stored decisions remain immutable. Presentation rule
  `evidence-led-assessment/v5` interprets historical marker-led results as
  **inconclusive**, explains the retired rule and assigns no confidence. JSON
  retains the original saved decision for audit, with the current interpretation
  in `customer_assessment`. Customers should use that interpretation rather than
  treating a historical machine code as a current endorsement.
- Browser observations and newly generated PDF findings, provenance tables and
  technical ledgers distinguish an unverified marker from an AI conclusion.
  Previously issued PDFs are not silently rewritten.

## Verification and reproducibility

The [aggregate receipt](evaluation-panels/unverified-marker-policy-v1-result.json)
records the four paired outcomes and two actual PDF render checks. Fresh and
historical forensic PDFs contain the marker observation and scope limitation,
without the retired forced-score claim. Their canonical result bytes and frozen
snapshots remained unchanged during rendering. The 14-page historical and
13-page fresh QA reports were rendered to images; the changed summary, finding,
provenance and ledger pages were visually checked. Private QA files and per-run
application databases are not part of this release.

`backend/scripts/probe_unverified_markers_v1.py` prepares deterministic authored
inputs, then runs `before` or `after` exactly once into a new phase directory.
For baseline reproduction, use the released probe script with a checkout of the
baseline commit; run the corrected phase against the same prepared folder.
`backend/scripts/review_unverified_marker_policy_v1.py` verifies paired source
hashes, recorded outcomes, continued model-check attempts and immutable results,
then renders actual reports. It refuses an existing output directory.

Regression checks also exercise the real packaged image fuser, retirement of a
synthetic marker-dependent calibration, image/video/audio documentary gates,
historical interpretation across all four modalities, both JSON export schema
versions, browser/backend parity and public-report disclosures. These are code
and artifact checks by the development process, **not outside human validation**.

Validation: **375 selected backend tests and 90 browser unit/component tests
passed**, as did the production build, type checking, Ruff and diff whitespace
checks. The additional tests are policy and reporting regressions, not new model
accuracy measurements.

No serving weights, speech calibration, Stripe price or deployment were changed.
GCP remains deferred. Broader detection improvements and real-speech/background
watermark qualification remain separate evaluation tasks.
