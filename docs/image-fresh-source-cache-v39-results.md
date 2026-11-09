# Fresh reference pixels: 161 photos in 155 candidate-only components

The prospectively selected 163-reference acquisition and pixel review is complete.
**161 native JPEG references qualified, forming 155 candidate-only components**
with no link to the 705 prior inventory records found by the fixed grouping
rules. Two static JPEG endpoints returned 404; both failed cases and available
response bodies are retained without replacement.

These are references for a future controlled-generation study. This result does
not measure AI detection accuracy or establish authenticated camera origin.

| Stage | Records |
| --- | ---: |
| Original metadata selection | 256 |
| Unresolved metadata, retained in earlier denominator | 67 |
| Metadata matches with a known creator link, excluded | 26 |
| Frozen reference-JPEG requests | 163 |
| Qualified native JPEGs | 161 |
| Unavailable JPEG endpoints | 2 |
| Qualified photos in components linked to prior inventory | 0 |
| Candidate-only components | 155 |

## What was checked

The [protocol and exact plan commitment](image-fresh-source-cache-v39-protocol.md)
were published and remotely verified before the first image request, in commit
`5c4c1b10effae33ab30b208aa95c5b98bd9949f7`. Each source was requested once in
the frozen metadata order. The 161 successful source records reproduce exact
encoded-byte and native-RGB hashes, dimensions, upright orientation and the
fixed non-neural thumbnail sketch from the saved file.

The complete graph contains 866 records: all 705 prior records plus 161 qualified
new references. It uses the unchanged v22 source-ID, canonical-creator,
exact-RGB and informative-thumbnail rules, including full transitive components.
The 161 new references reduce to 155 groups through known creator links. No
cross-prior link was found under those rules. Creator aliases and transformed
duplicates can still be missed; this does not prove 155 independent people.

The retained new licence snapshots are 91 CC BY 2.0 and 70 CC BY-SA 2.0, matched
to current provider attribution and historical source identity in the prior
[metadata review](image-fresh-attribution-v38-results.md). Those snapshots do
not clear an unaudited generator or authenticate the depicted event.

## Local execution evidence

All six acquisition phases and the separate final readback/grouping phase
completed with exit zero, no sampling error and no observed resource crossing.
Acquisition phases lasted 2.14–12.27 seconds; the largest sampled worker-tree
RSS was 135,217,152 bytes. The final review lasted 1.34 seconds with sampled RSS
130,531,328 bytes. Sampled RSS is a lower bound, not a guaranteed peak.

The 163 records account for 28,659,626 encoded response bytes kept in the private
Seagate cache. Failed responses are included. Source photos, identifiers,
profiles, source pairings and sketches were not published. Fifty-three focused
local tests and Ruff passed before the protocol freeze; the actual completed
readback additionally verified all qualified saved source pixels.

The [aggregate receipt](evaluation-panels/image-fresh-source-cache-v39-results.json)
binds the plan, graph, summary, phase profiles and final monitor by hash. No
workflow ran and no outside person was contacted.

## Next admission step

Keep these references out of detector fitting and scoring until generator
component licences and bounded authored hardware/geometry probes are reviewed.
Then prospectively freeze the controlled study's source roles, generators,
masks, prompts, seeds, matched ordinary edits, formats and failure denominators.
Do not select generated examples by visual quality or detector success.

No neural model, generator, calibration or detector evaluation ran in this
collection; no source role was assigned. The current v26 training and held-out
roles remain unchanged. Unknown earlier creators, aliases, transformed
duplicates and foundation-model pretraining overlap remain limitations. There
is no broader accuracy or commercial-serving admission from these counts.
