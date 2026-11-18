# Remote sentence similarity and an exact overlap with a 2020 revision

An independent local test asked whether sentences that fit poorly with
other parts of the same document could reveal short AI rewrites. One
variant compared a sentence with its three most similar peers outside its
immediate context. The other compared it with all eligible remote peers.
Both used the same pinned, Apache-2.0 sentence encoder, with all learned
parameters frozen and no earlier draft supplied at inference. The design
draws on research into sentence relationships in mixed-authorship text.
[SenFlow study](https://arxiv.org/abs/2606.18946)

## A source label that the text alone cannot resolve

One later local rewrite reproduced the same sentence and complete analyzed
text as a revision published by *The Turing Way* in November 2020. The
archive check verified the parent and child Git objects, their chronology,
the shared changed sentence and whole-document hash. The local generation
record used the parent excerpt as input. It provides an example of a
different observed process yielding the same text. Model pretraining
exposure and the tools used by the historical editor are unknown.
[Publisher parent commit](https://github.com/the-turing-way/the-turing-way/commit/e67da5d46cbcc51b4b1c06aa07bced2edbc43166)
· [Publisher revision](https://github.com/the-turing-way/the-turing-way/commit/60833fbc5680a60dab793fca8cffd3e0979fbf5d)
· [Source and digest receipt](evaluation-panels/writing-turing-origin-overlap-2026-11-18.json)

The audit found the overlap before measuring the new sentence scores and
excluded the whole source group. The corrected denominators for the earlier
study appear alongside its historical test counts. [Earlier study](writing-turing-editorial-history-2026-11-18.md#local-passage-results)

## Held results

On 165 held files with 68 labeled edits, both variants kept the existing
2/68 useful locations, added two misplaced highlights on edited files,
and flagged four additional ordinary files across three source groups.
Before applying a cutoff, each method ranked the right passage first for
2/68 labels. Uniform selection among eligible passages would be expected
to cover about 6.5 labels. Sixty-five labels had an eligible range. The
comparison supports keeping both variants experimental.

| Ordinary files with a highlighted passage | Current | With nearest three | With all remote peers |
| --- | ---: | ---: | ---: |
| Untouched | 13/60 | 15/60 | 15/60 |
| Ordinary substitutions | 9/41 | 11/41 | 10/41 |
| Dated editorial revisions | 1/14 | 1/14 | 2/14 |

A second native run reproduced the numerical inputs and results byte for
byte. The public receipt includes anonymous source-group aliases, hashes,
ranges and the scores needed to recalculate the fixed review points and
outcomes. The source prose and embeddings stay local.
[Numeric receipt](evaluation-panels/writing-remote-semantic-transfer-2026-11-18.json.gz)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_writing_remote_semantic \
  docs/evaluation-panels/writing-remote-semantic-transfer-2026-11-18.json.gz
```
