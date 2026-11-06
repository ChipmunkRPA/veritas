# Pre-purchase analysis limitations

Implemented locally on 2026-11-05. This change adds an explicit acknowledgment before payment; it does not change scoring, calibration, the USD 9.99 price, refund terms or the requirement to confirm payment before inference.

The customer sees that automated checks can miss deepfakes, flag genuine material, or return an inconclusive result with no usable confidence or reliable conclusion. Payment purchases the analysis and available evidence rather than a guaranteed detection or answer. The standard SaaS report is preliminary and not for reliance as court evidence; privately researched expert work requires a separate signed engagement.

Document uploads also disclose the current known limitation: article checks provide inspection evidence without a calibrated whole-article authorship conclusion. The separate, limited email-body check is distinguished from that article scope.

The acknowledgment is unchecked initially and resets when the upload, text, media kind, profile or notice version/hash changes. The server rejects missing acceptance and incorrect notice versions/hashes before staging a new order or creating a Stripe checkout. A client-side disabled button alone is insufficient.

`purchase-risk/v1` has a canonical text hash. The server stores the exact disclosure, accepted flag, server acceptance timestamp and authenticated principal with the order in the nullable `purchase_acknowledgement` JSON column introduced by migration `0026`. Receipt data is server-authored. Existing accepted receipts are not overwritten on retry. A completed historical order cannot be given a retrospective acceptance through the resume route.

New Stripe sessions repeat a concise warning in `custom_text[submit][message]`. [Stripe's hosted Checkout custom-text contract](https://docs.stripe.com/api/checkout/sessions/create) supports this field. No upload names or source content are included. Orders created before this feature require acceptance before resuming through this app; their original processor parameters remain unchanged to preserve unknown-outcome retries and avoid duplicate payments. Already-paid historical orders remain fulfillable under their original verified payment receipt.

Verification uses isolated databases/authentication/queues and test-double Stripe responses, including rejection without acknowledgment, wrong version/hash, timestamped receipts, duplicate submission, cross-owner access, old-order resume, retrospective-acceptance refusal, fixed price and signed-payment fulfillment. Browser tests cover unchecked state, blocked continuation, source-change reset and resumed-order acknowledgment. No real charge is used for these checks.

## Evidence presentation

The subsequent evidence-led presentation is implemented as `evidence-led-assessment/v4`:

- Remove the advertised percentage cutoff; keep internal calibrated decision rules and abstention safeguards.
- For each finding, distinguish the observable fact from its interpretation and link the source crop, frame/time range, waveform/audio interval or quoted passage.
- Use AI-leaning, original-leaning, inconclusive and context-only states. Lack of an AI flag is not proof of original content.
- Show numerical confidence only when its scope has a defensible calibration. Do not turn a raw detector response, metadata anomaly or waveform shape into an AI probability.
- Preserve the existing calibrated combination of applicable evidence and its exclusions. Do not average confidence numbers. Where that combination is unsupported, show no overall confidence and explain the missing coverage.

The public fixed-cutoff claim has been removed. Internal scoring and abstention safeguards remain unchanged. The pre-purchase warning and its acceptance receipt remain required before payment and analysis. Dropping a public cutoff or adding disclaimers does not resolve the previously measured localized-edit and face-swap misses.
