# Fresh image-source metadata: completed collection and corrected readout

The fixed 256-record collection is complete. **163 photos have matching current
provider attribution/licences and no identified creator-key link to the existing
705-record inventory**, across 157 distinct profile keys. These are candidates
for later pixel review, not 157 proven independent test subjects or a detector
accuracy result.

## Complete denominator

| Outcome | Records |
| --- | ---: |
| Selected before current-provider requests | 256 |
| Current photo, owner and recorded licence match | 189 |
| Unresolved, retained without replacement | 67 |
| Matched records linked to known creator keys | 26 |
| Matched records without an identified creator-key link | 163 |

The 189 matches comprise 111 CC BY 2.0 and 78 CC BY-SA 2.0 records. The 67
unresolved cases are 41 unavailable endpoints, 15 current/historical licence-ID
mismatches, nine responses failing the required Flickr photo schema and two
static-source identity mismatches. A licence mismatch does not by itself mean
that permission was revoked; it fails this exact-correspondence protocol.

All eight 32-request phases completed under their resource monitors with exit
zero, no sampling error and no resource crossing. Phase durations were
8.51–9.32 seconds; the largest sampled worker RSS was 52,248,576 bytes, not a
guaranteed peak. The method and exact private-plan commitment were
[published and remotely verified before requests](image-fresh-attribution-v37-protocol.md)
in commit `2987384b93aa4e7d294da5162c68bf13852ec71c`. No workflow ran.

## Explicit readout correction

The original v37 summary failed after collection: it passed already-normalized
creator keys from the v22/v25/v35 source graph to a helper expecting full profile
URLs. The actual source-record constructor stores `author_key(profile)`.

The v38 successor validates and uses those stored keys directly, while applying
the original URL normalization to newly received profile URLs. It re-verifies all
256 source/response bindings, retained bytes, completed phase journals and
successful monitor receipts. It does not repeat requests, replace sources,
modify responses, alter licence rules or change earlier code. The failed v37
summary and the earlier failed first planning monitor remain retained as failed.

The separate corrected readout completed in 0.54 seconds with exit zero and no
monitor error or resource crossing. Its sampled worker RSS lower bound was
48,234,496 bytes. Forty-seven focused local tests pass, including regression
cases for pre-normalized creator keys, linked owners, unknown creators, schema
errors, failed-request denominators and exact licence/photo correspondence.
Ruff passes.

## What this enables, and what remains

The next step is to verify the candidate source pixels, orientation and complete
exact/near-duplicate components against the existing inventories. Creator aliases,
14 prior unknown creator identities and unknown pretraining overlap remain
limitations. A canonical profile key is not a verified person or camera source.

Only after pixel/group review and generator component/hardware qualification
should a controlled multi-generator study freeze roles, masks, prompts, seeds,
matched ordinary edits, formats and failure denominators. No image, model weight
or neural output was acquired here; no training/evaluation role was assigned.
The running v26 study, held-out roles, serving defaults and confidence claims are
unchanged. No outside person was contacted and no cloud resource was used.

The [aggregate receipt](evaluation-panels/image-fresh-attribution-v38.json)
contains counts, hashes and resource observations. Provider response bodies,
photo IDs, creator profiles and source pairings remain private.
