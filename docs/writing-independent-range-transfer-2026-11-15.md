# A separate classifier for selecting short rewritten passages

The existing word-range head often missed short rewrites even when it
scored the correct sentence. This comparison measures two separate
decisions. One is which sentence to inspect. The other is which words
within that sentence overlap the known rewrite. The test asks whether a
separately trained, small classifier improves the first decision enough
to improve the final evidence range.
[Earlier word-range comparison](writing-range-context-2026-11-15.md)

## Fixed method and inputs

The 68M-parameter Ettin detector runs locally in INT8 on the CPU. Its
publisher grants MIT terms for the pinned model files. The tokenizer and
model are loaded from disk and use no publisher Python code. The trial
calculates the raw AI-minus-human logit on each of three interior
sentences, keeping its original Unicode range. It applies neither the
publisher's whole-document cutoff nor its probability interpretation.
The narrower task uses this project's separate ordinary points.
[Pinned model and license](https://huggingface.co/Deckard618/textdet/tree/8b677e1bd3bc6421ae710352cda13ca65502147b)
· [Ettin encoder paper](https://arxiv.org/abs/2507.11412)

On each sentence, compare the independent score, its margin over the
other two interior sentences, and each of those signals blended with
the frozen paired range head. The two blended arms use 75% of the paired
rank and 25% of the independent rank. The larger weight anchors the
selection to the check that actually proposes a range of words. This
borrows the idea of a bounded correction from NormDetect. It uses its
own sentence scoring and ordinary-source calibration.
[Anchored calibration research](https://arxiv.org/html/2610.41459v1#S3.SS6)

All arms use the same 144 calibration works, divided into reports, news
and abstracts. Each component is ranked against ordinary works of the
same genre and sentence position. The calibration of each arm excludes
the work being ranked and uses the strongest eligible sentence per
work. The two points use the same tighter and wider ordinary-work
budgets as the earlier study. They were computed before held inference.
A tie withholds the selection. For word overlap, any newly selected
sentence takes its existing word proposal unchanged. A finding can
select the right sentence while highlighting too few of the changed
words. A useful range covers at least half of the known changed words
and has at least 0.30 word intersection over union.

The held panels contain 309 rewrite exemplars and 319 ordinary files.
They reuse the prior source-disjoint report replications and news and
abstract cohorts. The eLife panels include real dated revisions and a
separate set of original and same-work replacement excerpts. No held
work appears in the 144-work point selection roster. The new
classifier's publisher lists RAID, MAGE and DetectRL-X for fine-tuning.
Overlap with source works in its pretraining or related corpora is
unknown. The fixed paired range head was already evaluated on these
project panels. [Publisher training disclosure](https://huggingface.co/Deckard618/textdet/blob/8b677e1bd3bc6421ae710352cda13ca65502147b/README.md#training)
· [Matched report and genre source commitments](writing-paired-ranges-2026-11-15.md)
· [Ordinary journal revisions](writing-edit-occurrence-transfer-2026-11-15.md#new-genuine-revision-comparison)

## Held results

Each cell gives correct word ranges followed by ordinary files flagged
at the wider fixed point. The complete receipt also records tighter
points, correct and wrong sentence locations, and paired gains and losses.

| Panel | Paired word baseline | Independent sentence score | Independent margin | Blend with independent score | Blend with independent margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reports | 3/93 · 16/93 | 2/93 · 30/93 | 3/93 · 25/93 | 4/93 · 16/93 | 3/93 · 20/93 |
| Separate reports | 0/48 · 8/48 | 1/48 · 20/48 | 0/48 · 20/48 | 0/48 · 12/48 | 0/48 · 10/48 |
| News | 0/56 · 8/56 | 1/56 · 29/56 | 1/56 · 17/56 | 2/56 · 14/56 | 0/56 · 12/56 |
| Retained abstracts | 1/60 · 25/60 | 1/60 · 16/60 | 4/60 · 25/60 | 2/60 · 14/60 | 3/60 · 20/60 |
| Additional abstracts | 2/28 · 8/28 | 0/28 · 9/28 | 0/28 · 10/28 | 1/28 · 3/28 | 1/28 · 6/28 |
| eLife body excerpts | 0/24 · 19/30 | 0/24 · 2/30 | 1/24 · 10/30 | 0/24 · 13/30 | 0/24 · 14/30 |
| Dated journal revisions | — · 2/4 | — · 4/4 | — · 1/4 | — · 2/4 | — · 2/4 |

On the separate reports, the independent score selected the correct
sentence in ten of 48 files, compared with three for the paired word
check. Once supplied the correct sentence as an evaluation-only
diagnostic, the existing head's word proposal reached useful overlap
in only four of 48 cases. In two of those 48 files the prior word head
could supply no word range. Checking the correct sentence more often
therefore has limited benefit without better word proposals.

The blend added one useful report location with the same number of
ordinary flags. The same sentence also flagged in its matched original.
The two news locations from that blend did not also occur in the
matched originals, and six additional ordinary news files flagged.
The retained abstract gains did not transfer cleanly to the additional
works. The sole useful eLife result also selected the same sentence in
a matched ordinary input. These directions remain experimental.

## Independent replay

The [public receipt](evaluation-panels/writing-independent-range-transfer-2026-11-15.json.gz)
preserves three raw scores per source hash and binds the existing source
and word-label receipts. The portable reviewer recomputes points, held
outcomes and the known-sentence diagnostic without weights or source
prose. Python 3.12 and 3.14 produced identical review output. The inputs
can also be independently re-scored with ONNX Runtime 1.24.3 and the
hashed model and tokenizer. A second native run reproduced all 2,316
sentence responses and the receipt byte for byte.

```sh
export PYTHONPATH=backend
module=scripts.writing_independent_range_transfer
python3 -B -S -m "$module" review \
  docs/evaluation-panels/writing-independent-range-transfer-2026-11-15.json.gz
```

For native replay, use the pinned [report and news/abstract rosters](writing-context-response-words-2026-11-15.md#independent-replay),
the [dated and eLife source plans](writing-range-context-2026-11-15.md#replay),
the two pinned abstract shards and the sealed [report replication](writing-paired-ranges-2026-11-15.md#independent-replay).
Save the pinned `ettin-68m` model, tokenizer and config with the repository
`LICENSE` from the model release. Point `PYTHONPATH` at a local ONNX
Runtime installation when it is outside the backend environment.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
backend/.venv/bin/python -B -m "$module" capture "$run/independent.json.gz" \
  --assets "$model" --reports "$report_roster" --matched "$matched_roster" \
  --additional "$additional_roster" --dated "$dated_revisions" --elife "$elife_plan" \
  --fresh "$sealed_replication" --gemini "$gemini_shard" --gpt "$gpt_shard"
```
