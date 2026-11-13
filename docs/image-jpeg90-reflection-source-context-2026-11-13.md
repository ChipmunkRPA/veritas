# Source context for JPEG90 reflection flags

The [JPEG90 reflection study](image-regional-jpeg90-reflection-2026-11-13.md)
kept all four useful locations from its direct JPEG90 check and also kept
ten ordinary control observations. This follow-up checks where those
observations fall on an exact earlier copy of each source. It reuses all 88
held deliveries from that study. No operating point or serving rule changes.

Reconstruct the submitted JPEG90, authored truth, and earlier JPEG90 from
the held source roster. Match all delivery and truth commitments. Rerun the
direct native checkpoint only on the 25 deliveries with recorded components.
Verify each native mask and component digest, pixel area, and overlap with
the authored truth before comparing pixels. The hard reflection decisions
come from their unchanged ledger. Measure exact decoded RGB change and the
existing earlier-copy comparison’s difference mask at those same component
pixels. [Reflection ledger](evaluation-panels/image-regional-jpeg90-reflection-2026-11-13.json)
· [Held delivery ledger](evaluation-panels/image-regional-jpeg90-quality-confirmation-2026-11-13.json)

The ten ordinary reflected views came from two photos, with 13 components.
Nine components were pixel-identical to their earlier copies. One more
differed at 18 of 38,132 pixels. The two resize controls had RGB differences
inside their components and zero support from the existing earlier-copy
difference check. Twelve of the 13 components had no support from that
paired check. The one supported component had 490 pixels in a known
conventional local paste. Nine of the ten views had no supported component.

The existing earlier-copy comparison assessed all 88 pairs. It located all
22 authored insertions or removals under the frozen overlap definition,
including the 12 LaMa cases. It also showed differences in 31 of 66 ordinary
controls, including all 11 local tone adjustments and all 11 conventional
pastes. The 11 untouched copies had no observation. A paired comparison
answers where these two matched files differ. The earlier input in this
diagnostic is the exact same-framing authored earlier copy. An arbitrary
earlier file, a different camera pipeline, or an unknown delivery history
requires a separate evaluation. The result supplies no basis for a
single-upload JPEG90 expansion or an authorship conclusion.

## Replay

The standard-library reader checks all anonymous scalar records, mask
commitments, aggregate counts, and hashes of both input ledgers. Native
replay requires the same three private source roster paths, checkpoint, and
backend dependencies described in the [held study](image-regional-jpeg90-quality-agreement-2026-11-13.md#reproduce).
The receipt contains no images or model weights. [Source-context receipt](evaluation-panels/image-jpeg90-reflection-source-context-2026-11-13.json)

```sh
python3 -B -IS backend/scripts/review_image_jpeg90_reflection_source_context.py review
PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1 \
  backend/.venv/bin/python -B -m scripts.review_image_jpeg90_reflection_source_context native \
  --main-roster "$main_roster" --external-roster "$external_roster" \
  --confirmation-roster "$confirmation_roster"
```
