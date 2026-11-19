# Can a writing highlight survive removal of either neighbor?

The two experimental rules located zero of three verified sentence rewrites
and raised zero alerts across eight ordinary work groups. Both ranked the
same true edit first before applying the fixed operating points. In the
other two rewritten files, both ranked a distant sentence first. The
application keeps its current passage check and confidence scoring.
[Anonymous measurements](evaluation-panels/writing-retained-neighbor-2026-11-19.json)

## Matched sources and comparison

The test used English PostgreSQL and Kubernetes documentation from dated
2020–2022 Git snapshots. Each pair has a confirmed parent and child,
12 aligned sentences and exactly one changed sentence. The intake excluded
recorded prior chapter paths and matching normalized 12-word fragments.
Five separate work groups supplied development controls. The held set
contained four PostgreSQL and four Kubernetes works, each with its untouched
excerpt and actual archived revision. The dated records establish the
wording changes. Checkpoint exposure to the sources and the contributors'
writing tools are unknown. [PostgreSQL documentation license](https://www.postgresql.org/about/licence/) · [Pinned Kubernetes website license](https://github.com/kubernetes/website/blob/e20b8746f2444ae9fe58f49fd0f0acbfbabbb938/LICENSE)

Each held work supplied two fixed prompts to its assigned local generator.
Nine of 16 attempts met the structural checks. Four changed a technical
claim, two remained unclear and three preserved the claims. The seven
structural exclusions included four format errors, one changed sentence
boundary, one unsuitable degree of wording change and one exact match to
the dated ordinary revision. The three admitted rewrites came from distinct
works spanning both publishers and both generator families. All 16 ordinary
files stayed in the comparison, including those whose generated version was
withheld. The factual and positional review closed before detector scoring.

The new rule scores the target sentence alone, with only its left neighbor,
and with only its right neighbor. It ranks the weakest of those three
length-adjusted readings. A sentence-only ranking and the app's optional
passage-interval result provide the comparisons. Every proposed highlight
requires a unique strongest sentence. Each point was fixed on whole-work
maxima from ordinary development pairs before held scoring. These local
readings share a checkpoint and provide no independent confidence votes.
[Earlier factorial study](writing-neighbor-factorial-2026-11-19.md) · [Sentence localization research](https://aclanthology.org/2024.findings-acl.495/)

| Held check | Exact rewrite sites / 3 | Wrong highlights | Ordinary groups flagged / 8 |
| --- | ---: | ---: | ---: |
| Current optional passage check | 0 | 0 | 0 |
| Sentence only | 0 | 0 | 0 |
| Sentence and either remaining neighbor | 0 | 0 | 0 |

All three rewrites and all ordinary files were assessable. With three
eligible edited works, zero alerts leave practical detection and false-alarm
rates uncertain. The withheld attempts show how easily a technically fluent
rewrite can change a required term or qualifier.

## Recount

The anonymous receipt carries sentence identities, length counts, selection
ranges and scalar readings. The standard-library script replays both
candidate rankings, their points, held matches and generation exclusions.
The local run measured 525 distinct inputs on one CPU thread in about 36
seconds after loading the classifier. The exact source prose, contributor
details and generated text stay in the private evaluation workspace.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_writing_retained_neighbor \
  docs/evaluation-panels/writing-retained-neighbor-2026-11-19.json
```
