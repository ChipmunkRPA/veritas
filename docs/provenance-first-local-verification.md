# Provenance-first local verification

7 November 2026. This is software validation of documentary verification and
customer disclosures, not a provider-watermark accuracy benchmark or a new
statistical model admission.

## Implemented

- Image, video and supported audio pipelines check trusted, valid, asset-bound
  Content Credentials before model acquisition. An explicit signed AI declaration
  stops subsequent detection and records the skipped checks. Unsupported,
  absent, stripped, untrusted or invalid credentials continue detection.
- Audio has a model-free credential adapter for WAV/M4A. Its signed-declaration
  path works in both speech modes without forming a statistical voice confidence.
- The website, pre-purchase acknowledgement, Stripe message and report/export
  policy explain uncertainty, misses, false alarms and inconclusive outcomes.
  Private investigation requires a separate engagement; it guarantees neither
  a conclusion nor court admissibility. Contact: sales@chipmunkrpa.com.
- New purchases require the current disclosure. Existing paid orders retain
  their original valid acknowledgement. USD9.99 prepayment is preserved.
- Summary PDFs no longer repeat the private-engagement paragraph or force an
  otherwise empty evidence chapter. An explicit missing-evidence warning remains.

## Local checks

- 345 selected backend tests pass across the provenance gate, actual signed
  media, historical/current purchase consent, payment-before-analysis worker
  gate, customer assessments, face checks, evidence exports and report guards.
- Actual SDK verification uses authored PNG, WAV and MP4 files signed by a
  locally generated test authority. Pixel tampering and metadata stripping
  cannot activate the shortcut. The test authority is never added to the
  application's production trust list.
- Twelve isolated full-pipeline cases reach published results with model leases
  forbidden: image, video, legacy audio and candidate audio, each in quick,
  full and deep mode. This does not test real provider account access.
- 71 selected frontend tests, type checking and the production build pass.
- A summary video PDF was rendered, its final page visually reviewed, and its
  vocabulary/internal-identifier lints pass. It fits on one page with the
  disclosure, findings and missing-evidence warning intact.
- Serving model asset records and the local speech calibration bytes are
  unchanged. The speech calibration SHA-256 remains
  `912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf`.

## Limits and next work

The [provider review](provider-provenance-coverage.md) covers OpenAI, Anthropic,
Google/Gemini, Meta/Muse, xAI/Grok, MiniMax and DeepSeek. Private watermark keys
or verification APIs are not simulated. No customer files were uploaded to
external verifiers. Offline credential checking has trust-snapshot, revocation,
remote-manifest and container limitations. A signed declaration does not locate
each edited region or establish the entire history of a file.

The [recent-method review](recent-methods-review-2026-11.md) informs further
development; reading a paper does not establish an accuracy gain. Real provider
fixtures and supported verifier integrations, structured labeling metadata,
new independently implemented methods and fresh controls remain work. The
separate image holdout remains unscored. No statistical confidence threshold,
model weights, serving calibration or GCP deployment changed in this release.
