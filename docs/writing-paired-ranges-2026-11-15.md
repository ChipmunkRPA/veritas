# Training writing ranges against matched original passages

The earlier range check learned which proposed word spans overlapped a
known rewrite. The paired candidate also learns to give a useful range
a higher response than the strongest ranges in the corresponding untouched
sentence. This tests whether a model can use the same subject and surrounding
text to learn a difference that survives on new works. It reads a single
excerpt at inference. Source-aligned supervision and matched editing regimes
motivate the comparison. [MixDetect word-level study](https://arxiv.org/abs/2609.32625)
· [ARB matched rewriting benchmark](https://arxiv.org/abs/2607.29539)

The same pinned Apache-2.0 encoder, range geometry, eight-epoch schedule
and development work roster were used for both range heads. Of the 652
eligible fit files, 259 offer at least one useful generated-revision range.
For those files the new loss compares the mean of the useful ranges with the
mean of the four highest responses in the matched ordinary sentence, with a
margin of one. All calibrated points use separate ordinary works. A useful
range covers at least half of the changed words and reaches at least 0.30
word intersection over union. The comparison uses project code and the
previously pinned data and weights. [Original protocol and reuse links](writing-supervised-ranges-2026-11-15.md)

## Held results

The first panels repeat the earlier file roster to isolate the effect of
training. The fresh replication uses 48 new report works, 16 per interior
sentence position, selected by source hash before scoring. It excludes the
previous report rosters, the 48 works in the earlier replication, and known
normalized 12-word overlaps. It contains 31 polishes, 15 paraphrases and
two compressions. Each original and edited version was scored on its own.
Both heads use their own frozen report points for this comparison.

| Held outcome | Earlier range head, tighter → wider | Paired range head, tighter → wider |
| --- | ---: | ---: |
| Local Qwen report rewrites located | 5 → 5 / 32 | 3 → 3 / 32 |
| Earlier GPT-5.4 report rewrites located | 0 → 0 / 60 | 0 → 0 / 60 |
| Earlier ordinary reports flagged | 8 → 19 / 93 | 2 → 16 / 93 |
| Fresh GPT-5.4 report rewrites located | 0 → 0 / 48 | 0 → 0 / 48 |
| Fresh ordinary reports flagged | 6 → 7 / 48 | 3 → 8 / 48 |
| News rewrites located | 0 → 0 / 56 | 0 → 0 / 56 |
| Ordinary news flagged | 10 → 18 / 56 | 7 → 8 / 56 |
| Abstract rewrites located | 1 → 2 / 60 | 1 → 1 / 60 |
| Ordinary abstracts flagged | 3 → 20 / 60 | 6 → 25 / 60 |

The tighter paired point gained one Qwen location and lost three. It
cleared six ordinary flags in the earlier report panel and three in the
fresh report panel. At the wider point it flagged both versions of one
of the genuine journal revisions. The earlier range head flagged none
of the four genuine journal excerpts.

On the fresh panel, the context word check located one edit at its wider
point, with nine ordinary flags. Its jointly calibrated decision with the
paired head located that same edit with eight ordinary flags. The matched
untouched file also flagged the corresponding sentence. Combining the
two range heads located zero edits. Among 42 assessed fresh cases with
changed words, 30 had a proposed range positioned well enough to count.
Giving the paired detector the exact target sentence raised the count to
four useful top-ranked ranges. The result points to weak ranking within
the edited sentence as well as across sentences. The paired head stays
in evaluation.

## Independent replay

The receipts retain source and model commitments, native offsets, assessed
coverage, score cutoffs and paired outcomes. Source prose and weights are
kept out of the public receipts. Two local CPU fits produced byte-identical
weights and report receipts, and both fits reproduced the same fresh
replication receipt. The numeric outcomes replayed identically with
Python 3.12 and 3.14 without model libraries.

```sh
export PYTHONPATH=backend
panels=docs/evaluation-panels
report="$panels/writing-paired-ranges-report-2026-11-15.json.gz"
python3 -B -S -m scripts.writing_supervised_ranges review "$report"
python3 -B -S -m scripts.writing_supervised_ranges review \
  "$panels/writing-paired-ranges-transfer-2026-11-15.json.gz" --report "$report"
python3 -B -S -m scripts.writing_range_report_replication review \
  "$panels/writing-paired-ranges-fresh-2026-11-15.json.gz" --report "$report"
```

For native replay, use the assets and roster reconstruction in the
[earlier range protocol](writing-supervised-ranges-2026-11-15.md#independent-replay).
Add `--paired-margin` to the `report` command and use its output as the
`--fitted` directory for both later commands. For the fresh replication,
also pass `--unpaired-head` with the earlier `range.safetensors`.
