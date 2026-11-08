# Real provider verification: qualification v1

The documented free Suno and Bria endpoints were actually called, after
[freezing the plan and source](provider-provenance-v1-protocol.md) in commit
`04ba62d2048ee60ce509032da39e9d9c9309c560`. Nine sequential requests completed
in 5.61 seconds, with no retries, customer uploads, model inference, paid
generation or account credentials. The
[aggregate result](evaluation-panels/provider-provenance-v1-result.json) is public;
raw media, individual pairings and response bodies remain private.

| Route | Requests / HTTP 200 | Reviewed-contract outcomes | Provider matches |
|---|---:|---|---:|
| Bria image verification | 7 / 7 | 6 completed; 1 inconclusive contract discrepancy | 0 |
| Suno credential verification | 2 / 2 | 2 completed, both `no_suno_provenance` | 0 |

## What this established

Both endpoints were reachable without API credentials. This is observed runtime
evidence, beyond reading their documentation. Authored pixels, an authored tone,
unsigned provider names and a copied credential did not verify as provider
provenance in these controls. A separate offline review checked all nine source,
request-body and reply commitments, replayed nine local SDK results, and verified
the three frozen code commitments. It made no additional network requests.

Both original images from Bria's historical public FIBO showcase contain explicit
AI declarations and matching active-file content hashes. Their signing
certificates are expired and untrusted in the current offline environment. Bria's
service also returned `Invalid`, with no detected invisible watermark. Neither
original verified. Thus a known provider-hosted AI example can lack a currently
accepted provenance result; a negative must continue detection, not clear the
file as original.

The credential transplanted onto authored pixels produced an actual content-hash
mismatch locally. That differs from the originals' certificate/trust failures.
The WebP request exposed a documentation/runtime discrepancy: the endpoint
returned an explicit false watermark boolean where its reviewed contract says
that field is omitted. The frozen parser retained this as **inconclusive**.
The parser was not changed afterwards to make the test pass.

The Suno tests are negative controls only. An observed public playback object was
not decodable as media and was excluded; it is not an authenticated song
download. No forbidden endpoint, encrypted playback protection or license gate
was bypassed. A genuine credential-bearing Suno positive remains untested.

## Concrete application corrections

- Separate matching content hashes from overall credential validity. Expired or
  untrusted credentials no longer imply that the pixels or sounds changed.
- Only an active-file hash match can satisfy the binding requirement; matching
  ingredient hashes cannot stand in for the uploaded file.
- Explain actual expiration, trust and binding facts in findings. Audio uses the
  same invalid-credential finding semantics as image/video.
- Correct website/PDF summaries that said credentials were not checked despite
  a recorded invalid result. Keep the original frozen record unchanged.
- Remove a raw structured-label dictionary and internal review references from
  PDF prose while preserving the source-label section and JSON evidence.

Two isolated full app analyses used the actual original provider files. Both
continued beyond provenance to applicable model checks; inference was deliberately
unavailable, and both withheld confidence and an original-file verdict. The
customer database and Stripe orders were untouched. The current trusted,
valid, asset-bound AI-declaration shortcut still works in the existing 30
signed-media pipeline cases; the failed provider records do not activate it.

344 selected backend tests and 12 frontend tests passed; TypeScript/production
build, Ruff and whitespace checks passed. A report from the actual isolated run
was rendered from its frozen snapshot. Its wording distinguishes a matching
content hash from expired credentials, records the actual invalid result, and
passes vocabulary/internal-reference checks with no refused assets. Three
stale customer-report expectations were first confirmed to fail on the baseline;
their updated checks preserve the user's removal of public threshold claims and
avoid inventing tendencies from incomplete signal metadata.

## Admission and remaining work

**No external provider route is enabled for customer analyses.** This small
contract study has no current valid Bria positive and no genuine Suno credential
positive. It does not establish detector-wide false-alarm/miss rates, current
product compatibility, or a statistical calibration gain. The WebP discrepancy
also needs resolution before that format is admitted. A false boolean must not
silently mean a check ran when it was unsupported.

Next qualification needs lawful current credential-bearing positives, relevant
unmarked controls and transformations, exact response/input association and
customer data controls. Existing local offline credentials continue to work
within their qualified scope. Model weights, the speech calibration, the
USD 9.99 prepayment requirement and GCP deployment remain unchanged.

Primary contracts: [Bria image verification](https://docs.bria.ai/visual-verification/verification/verify-image.md),
[Suno Credentials](https://suno.com/suno-credentials). The software/experiment
review was performed by this project; no outside human reviewer is claimed.
