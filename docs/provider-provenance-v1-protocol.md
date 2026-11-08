# Provider provenance qualification v1 — frozen protocol

This is a limited integration/contract study, not an accuracy calibration or a
serving admission. The code and private input plan are frozen before any
verification POST. Earlier work read documentation and inspected public samples
locally; it did not call a provider verifier.

## Inputs and prior observations

Nine cases are fixed: seven Bria image cases and two Suno audio controls. The
Bria set contains two original files from Bria's public FIBO showcase, one
metadata-stripped variant, one lossless WebP variant, authored pixels with a
copied credential, and authored pixels with/without an unsigned provider label.
The two Suno controls are an authored one-second tone with/without an unsigned
provider label. No customer media or paid generation is used.

The original Bria PNGs have explicit AI declarations, matching active content
hashes, and expired/untrusted signing credentials in this offline environment.
Those are preflight observations, not held-out accuracy results. One public
Suno playback object was not decodable as media; it is excluded. It is not an
authenticated original download, and no forbidden download endpoint, license
gate, encrypted-playback protection or account restriction is bypassed.
No genuine credential-bearing Suno positive is available in this plan.

The exact source bytes, transformations, request bodies and replies remain
private. Raw media, per-case pairings and provider reply bodies are not part of
the public package. Source examples are from the provider's public repository;
they are not relabeled as our own freely redistributable dataset.

## Frozen contract

Private plan SHA-256:
`1775e69325de94c55755aa2e232561d39af521616d3b7945ce23dd0bf25a5516`.

- Endpoint allowlist: Bria `POST /v2/image/verify` and Suno
  `POST /api/c2pa/detect` on the documented HTTPS origins.
- At most nine request attempts, sequential, no retries or redirects. Stop a
  route after HTTP 401, 403, 404 or 429; retain unattempted cases separately.
- Timeout 25 seconds per request; total study deadline 300 seconds. Input limit
  16 MiB; response limit 64 KiB. Every source and request body has a checksum.
- Freeze parser/runner/local verifier source hashes before execution. An
  existing execution log prevents silently repeating the experiment.
- Parse strict JSON, with duplicate/non-finite fields rejected. Bria's actual
  schema is nested under `result`, with a required request identifier. Preserve
  trusted/valid/invalid/absent credentials separately from watermark results.
  PNG/JPEG require the watermark field; GIF/WebP omit it as **not run**, never a
  negative. An inconsistent `verified` aggregate is inconclusive.
- Suno's three verdicts remain distinct. HTTP errors, pending replies,
  unrecognized schemas, missing applicable checks and transport failures are
  unavailable/inconclusive, never originality findings.
- All research responses have no confidence and no early-stop eligibility.
  Signed-by-provider and an explicit, valid, asset-bound AI declaration are
  distinct requirements. No production external route is enabled.

## Software change motivated by actual preflight

The prior report called any invalid credential evidence of a changed file, and
called a matching hash a mismatch when its certificate had expired. The update
separates active-file hash matching from overall credential validity and uses
plain explanations for expiration and signer-trust failures. Ingredient hashes
cannot stand in for the current file's binding. Trusted/valid state and an active
match remain required by `covers_bytes`; failures still prevent the shortcut.
This does not make expired credentials trusted or remove the offline trust limit.

## Evidence required after execution

Retain every attempted reply, status, latency, source/request/response checksum,
local credential result and explicit route-stop record. Reconstruct and verify
request commitments and summaries in a separate offline review. Publish
aggregate contract outcomes and limitations, without raw media/replies.
Run signed-media, binding-scope, report/export and prepayment regression checks.
Do not claim reliable provider coverage from this small case set or from software
tests. Real Suno positives, current Bria credentials, wider transformations and
customer data controls remain necessary before any external serving admission.

## Primary sources

- [Bria image verification contract](https://docs.bria.ai/visual-verification/verification/verify-image.md)
  and [OpenAPI schema](https://docs.bria.ai/_bundle/visual-verification.yaml).
- [Suno Credentials and developer API](https://suno.com/suno-credentials).
- [Bria public FIBO repository](https://github.com/bria-ai/FIBO), reviewed source
  tree `5a7269009c04173aeb570e34aa0a5d45aa1e6bc1`.
