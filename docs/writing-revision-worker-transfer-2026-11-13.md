# Published writing evidence on matched report revisions

OpAI-Bench supplies untouched government-report excerpts and successive AI
edits. A prior comparison highlighted 17 of 20 untouched reports using a
proxy based on the first six positive primary-classifier windows. This study
runs the Full document worker and measures the source ranges the app
actually selects for display. The app can show evidence from the paired
language-pattern check as well as the primary classifier. A selected
passage directs inspection. The document's overall conclusion is recorded
separately. [Benchmark paper](https://arxiv.org/abs/2606.06481) ·
[Pinned dataset](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9) ·
[Source audit and earlier proxy](writing-revision-trajectory-2026-11-13.md)

## Method

Each exact publisher excerpt is uploaded as a TXT file in an isolated app
with the same three-sentence local candidate settings and commercial model
policy used for the earlier writing tests. The real analysis worker runs on
Apple GPU. The billing test transport holds the job until a simulated
$9.99 USD payment clears. API and JSON packets agree. The check and runner
replay the source offsets and exact model-input hashes. The point of
comparison is the same native text and source geometry as the earlier
proxy. Model training overlap remains unknown.

Sixteen source-disjoint genuine reports set points before scoring any new
held worker output. At each candidate's second-highest genuine document
maximum, strictly higher responses can select up to six source ranges.
Two fixed ablations also retain either published packets with primary support
or the narrower subset where a primary positive coincides with the paired
ratio's exact sentence reduction.

- **Absolute response** ranks the raw classifier response. The highlight
  covers the full measured classifier context.
- **Within-document response** ranks classifier log odds above that document's
  median. The highlight covers the full measured classifier context.
- **Within-document agreement** ranks the weaker of the two robustly scaled
  deviations. One check is the classifier log-odds shift above its median;
  the other is the token-level ratio shift below its median. Only an exact
  sentence reduction can be selected. The candidate uses the classifier's
  broader context to nominate a location and measures the ratio over that
  sentence. The underlying models overlap in family and their scores may
  depend on the same text features.

The twenty previously scored report groups were prospectively reordered by a
salted hash of their frozen source identifiers and split into five batches of
four complete seven-version trajectories. No worker outcome was used for
ordering. The three candidate selection rules and their points were locked
before held worker scoring. The earlier proxy had already consumed these
sources. The comparison measures app parity and transfer on known sources.

A strict location covers at least half of the labeled non-whitespace
characters in the version and has at least 0.30 intersection over union.
A separate measure asks whether the union of visible highlights covers half
of any one publisher-labeled region. The benchmark projects recorded edit
operations into text spans. Overlap with those spans measures localization
against the publisher annotations. Byte-identical versions retain their
operation history in a separate stratum. A release would require strict
locations missed by the app on at least three different reports across both
generator families, with at most two held genuine reports selected and no
contribution to overall confidence.

## Genuine calibration controls

All 16 genuine fit reports completed the Full analysis with all three
configured checks and an inconclusive document conclusion. All 16 had at
least one published inspection packet. The worker published 95 packets.
Fifty-two had no primary-classifier support at the inspection point. The
union of the visible ranges covered 15,798 non-whitespace characters across
the 16 reports. These controls set the candidate points and cannot establish
held performance. A Standard PDF from one control showed the same six
source ranges as the evidence API and JSON without checkpoint names.
[Calibration controls and PDF receipt](evaluation-panels/writing-revision-worker-fit-2026-11-13.json)

The Full worker's three specialized position and second-reading reviews had
no applicable input in the calibration controls. The benchmark supplies one
paragraph per excerpt, and the original roster requires at least six
sentences. The ordinary inspection packets in this study should not be
used as evidence that the short- or multi-paragraph checks fail on their
supported inputs.

## First completed held batch

The first four source groups supplied 28 deliveries. Four are publisher
originals, 22 have changed text, and two carry a publisher editing label
with text byte-identical to the original. Both identical deliveries returned
the same native responses and inspection positions as their matched originals.
All 28 document conclusions were inconclusive, and all three configured checks
completed. The worker published 168 packets, including 73 without primary
classifier support. These counts cover one of five planned batches.

| Inspection method | Genuine reports selected | Changed versions with any half-covered region | Strict changed-version locations |
| --- | ---: | ---: | ---: |
| Prior primary proxy | 4/4 | 20/22 | 1/22 |
| Full worker, displayed packets | 4/4 | 19/22 | 0/22 |
| Full worker packets with primary support | 4/4 | 17/22 | 0/22 |
| Full worker packets with primary support and an exact ratio reduction | 4/4 | 12/22 | 0/22 |
| Genuine-fit absolute response | 1/4 | 3/22 | 0/22 |
| Genuine-fit within-document response | 1/4 | 1/22 | 0/22 |
| Genuine-fit within-document agreement | 1/4 | 1/22 | 0/22 |

On untouched originals, the Full worker highlighted 4,461 distinct
non-whitespace characters. Exact paired agreement left all four originals
with a highlight while reducing that total to 1,579. The fitted candidates
reduced the ordinary file count and missed most of the app's broader
publisher-annotation overlap. None improved strict localization. The three
fitted arms remain experimental. The remaining four batches are outside
this result. [First held batch and source-range receipt](evaluation-panels/writing-revision-worker-held-batch0-2026-11-13.json)

## Three completed held batches

The first 12 of 20 source groups supplied 84 deliveries. They comprised
12 originals, 61 changed versions and 11 byte-identical edited deliveries.
All 84 document conclusions were inconclusive. The worker published 494
inspection packets, including 283 without primary-classifier support.

| Inspection method | Genuine reports selected | Changed versions with any half-covered region | Strict changed-version locations |
| --- | ---: | ---: | ---: |
| Prior primary proxy | 11/12 | 50/61 | 2/61 |
| Full worker, displayed packets | 12/12 | 55/61 | 2/61 |
| Full worker packets with primary support | 11/12 | 47/61 | 1/61 |
| Full worker packets with primary support and an exact ratio reduction | 8/12 | 27/61 | 0/61 |
| Genuine-fit absolute response | 2/12 | 7/61 | 0/61 |
| Genuine-fit within-document response | 3/12 | 10/61 | 0/61 |
| Genuine-fit within-document agreement | 3/12 | 7/61 | 0/61 |

The broader app highlights touched more of the changed versions. They also
covered 13,160 non-whitespace characters in the genuine originals, up from
11,324 under the prior proxy. In the changed versions, 32.1% of the marked
characters fell inside a publisher edit span, compared with 39.9% under the
prior proxy. Both strict app locations appeared in the
earliest editing stage, one from each generator. The three frozen candidates
added zero strict locations. This readout covers three of five planned
batches. One interrupted batch recovered 17 saved rows and one already-paid
job before submitting its remaining ten deliveries. [Three-batch receipt](evaluation-panels/writing-revision-worker-held-batches0-2-2026-11-13.json)

## Replay

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_writing_revision_worker_transfer review \
  docs/evaluation-panels/writing-revision-worker-fit-2026-11-13.json \
  docs/evaluation-panels/writing-revision-trajectory-2026-11-13.json
PYTHONPATH=backend python3 -B -S -m scripts.review_writing_revision_worker_transfer review \
  docs/evaluation-panels/writing-revision-worker-held-batch0-2026-11-13.json \
  docs/evaluation-panels/writing-revision-trajectory-2026-11-13.json
PYTHONPATH=backend python3 -B -S -m scripts.review_writing_revision_worker_transfer review \
  docs/evaluation-panels/writing-revision-worker-held-batches0-2-2026-11-13.json \
  docs/evaluation-panels/writing-revision-trajectory-2026-11-13.json
```

Supply the private source roster reconstructed by the earlier benchmark.
The output directories may be placed on a disk with sufficient free space
and hardlink support. The full worker uses native checkpoint files and does
not contact Stripe or external inference providers.

```sh
export PYTHONPATH=backend
backend/.venv/bin/python -B -m scripts.writing_revision_worker_priority "$prior" "$batches"
backend/.venv/bin/python -B -m scripts.writing_revision_worker_transfer fit "$prior" "$fit" --device mps
python3 -B -S -m scripts.review_writing_revision_worker_transfer fit "$fit/worker.json" "$points"
backend/.venv/bin/python -B -m scripts.writing_revision_worker_transfer held "$prior" "$held0" \
  --device mps --fit-receipt "$fit/worker.json" --points "$points" --batches "$batches" --batch 0
```

Subsequent batches use `--batch 1` through `--batch 4` and new output
directories. Each worker records a per-file progress journal. An interrupted
held batch can be restarted with the same arguments and `--resume`, using
its existing isolated output directory. The runner verifies saved rows
against the authenticated API and JSON, then recovers any already-paid test
job before making a new simulated purchase. The public
reader accepts only consecutive completed batches and uses the prior
ledger's recorded native source overlaps. It checks the fit arithmetic,
identical-input behavior, and packet metrics from scalar and source-range
records.
