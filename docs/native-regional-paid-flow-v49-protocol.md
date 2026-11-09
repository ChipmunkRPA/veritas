# Native regional paid app flow, v49

Engineering verification with the full enabled commercial-policy image-check
set, real model weights and production HTTP, queue, detector, evidence and PDF
paths. Only database, storage and Stripe transport are isolated. No live Stripe
request, charge, deployment, outreach or GitHub Actions is allowed. This is not
another accuracy study; the fixed first v46 source was already consumed.

Freeze code, frontend build, runtime, available model/input bytes and enabled
checks before source inference. Five sequential isolated worker processes:
candidate mode on an authored 16-pixel scope input; then legacy edit, legacy
original, candidate edit and candidate original, using the same paired source.
All execute the full production image pipeline with every enabled check; missing,
skipped and failed checks must remain visible. No detector doubles, conditional
best-model choices, refitting or threshold adjustments.

Use CPU float32, two CPU threads, one queue worker and a 4096 MiB model-manager
budget. These deliberate smoke-test settings do not qualify the existing MPS
default or higher concurrency. Each monitored preparation/flow/seal step has
900 seconds, sampled worker-tree RSS at most 6 GiB, RAM/internal/external-disk
reserves 1.5 GiB and initial headroom 6 GiB RAM/3 GiB disk. Keep failed/guarded
attempts; no automatic retries, deletion, output overwrites or success relabeling.

An unpaid checkout and unpaid callback must create no analysis, job, weight load
or inference. A paid USD 9.99 simulated receipt must precede the actual worker's
model loading and native/legacy inference. The simulator validates the production
payment-entitlement path; it is not live Stripe settlement verification.

Require all enabled detector outcomes, native and legacy success on supported
PNG inputs, no original verdict for the undersized input, exactly unchanged
paired whole-file authorship numbers, and identical frozen crops/masks in API,
JSON export and PDF report snapshots. Verify every served presentation-asset
hash. Private model names and examination/sign-off controls must not appear in
customer API/JSON/PDF/ordinary UI. Native evidence must appear for the selected
edited input. The ordinary browser view must load original crops and toggle the
measured mask without script errors. Retain private screenshots and PDFs for
visual inspection; visual review is a separate requirement after automatic tests.

Only aggregate outcomes, code, whole-file hashes and limitations may be public.
Do not publish source photos/pairings, report crops/maps, credentials or weights.
Source-relative generative-edit localization still does not establish AI-specific
authorship against non-AI editing. No overall confidence contribution or local
default follows automatically from this flow check. Shared-pixel presentation,
further generator/control validation, other modality combinations, and existing
dependency security findings remain separate outstanding work.
