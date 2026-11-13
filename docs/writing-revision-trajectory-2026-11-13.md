# Writing test on an external public-report benchmark

OpAI-Bench records AI editing operations at several stages of a document's
revision history. We used two of its
publisher-held government-report shards to see whether isolated and adjacent
sentence checks can add locations the primary model misses on editing
trajectories. Both local model snapshots are Apache-2.0. This test uses an
outside research dataset and was executed by Chipmunk Verify. [OpAI-Bench paper](https://arxiv.org/abs/2606.06481) ·
[Pinned publisher release and data terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9) ·
[Government report source project and terms](https://gov-report-data.github.io/) ·
[Primary checkpoint](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615) ·
[Independent checkpoint](https://huggingface.co/rasbt/ai-text-detector-modernbert/tree/a6efd4bf6ecd852f332b2569ae0f98cae13b7985)

## Matched source roles

The publisher publishes character spans, sentence lists and inline tags.
Every admitted source version matched all three representations and retained
complete native character positions after the app's extraction and segmentation.
Twenty GAO source reports supplied one untouched publisher original and six
progressive variants per work. Two generator trajectories each contributed
polish, paraphrase and compression stages. Sixteen other untouched GAO works
from the published test shard set the experimental points, before any held
model scoring. Three of the first 23 shared source groups had unsupported
geometry and were excluded before inference. The selected original text and
normalized 12-word sequences were disjoint across these two roles. The GAO
URLs in the receipt are derived from the publisher's source IDs. The audit
reconstructs the benchmark records from the pinned shards and does not verify
that each published excerpt matches the text on its linked GAO page. Every
admitted version is a single paragraph, and checkpoint training overlap is
unknown.

The comparator is a **primary three-sentence proxy** that uses its first six
native context responses at the existing 0.50 research point. The customer app
can merge additional check findings and prioritize different evidence. The
first candidate measures the independent classifier on each isolated eligible
sentence. The second can also propose up to three consecutive sentences with
positive log odds in the same paragraph. Each candidate uses a separately
frozen genuine-work point and produces one proposal. The test also transfers
the independent sentence point from an earlier Kubernetes study without
refitting it. [Earlier independent sentence study](writing-multiblock-independent-2026-11-12.md)

## Results

Fifteen of 120 publisher-labeled edited versions are **byte-identical to their
matched original**. Eleven are Qwen polish, two are Gemini polish and two are
Qwen paraphrase. The frozen ledger retains these records. Their operation labels
describe the generation history even though the submitted text is identical.
The following changed-text readout excludes those 15 after the main run and
keeps all 20 ordinary controls.

| Outcome | Primary proxy | Proxy plus adjacent-sentence check |
| --- | ---: | ---: |
| Changed-text files with half of at least one annotated span highlighted | 83/105 | 84/105 |
| Changed-text files with one location covering half the total edit and at least 0.30 IoU | 4/105 | 4/105 |
| Untouched originals with any highlight | 17/20 | 17/20 |
| Highlighted non-whitespace characters in untouched originals | 17,429 | 17,704 |

The adjacent-sentence check added one half-covered Gemini compression work,
with no strict location gain. The separately fitted and transferred isolated
points added no new half-covered work at the frozen comparison. The high
ordinary proxy flag rate leaves little room for additional flagged works and
can hide increases in ordinary characters highlighted. The ledger preserves
both false-alarm measures and all per-stage outcomes. This evidence does not
support a new app check or confidence weight. Operation, stage and annotated
coverage change together in this panel.

## Replay

The local run assessed **3,042** exact native sentence positions with the two
installed checkpoints. The publisher-source replay reconstructed all **156**
records and validated every position, per-span overlap, source role and model
input hash against the original CSVs. This separate reader fixes a row-order
assumption in the first source verification attempt, which matched by position
rather than stable publisher record ID. Only scalar scores, source IDs, ranges,
hashes and arithmetic appear in the public ledger. [Decision ledger](evaluation-panels/writing-revision-trajectory-2026-11-13.json)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_writing_revision_trajectory_transfer \
  docs/evaluation-panels/writing-revision-trajectory-2026-11-13.json
```

For native source replay, place the pinned `README.md`, `croissant.json`,
`reports_gemini-2.5-flash.csv` and `reports_qwen3-8b.csv` in a scratch cache.
The CSV files are under `default/test` at the pinned revision. This check uses
local segmentation dependencies and no weights.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.review_writing_revision_trajectory_transfer \
  docs/evaluation-panels/writing-revision-trajectory-2026-11-13.json --cache "$publisher_cache"
```
