# When both writing checks flag a passage

The two writing checks can flag the same navigation sentence while
measuring different amounts of text. The predictability ratio can
reduce its result to the tokens of one sentence. The passage classifier
can flag a wider input that includes it. New reports explain the
difference beside the source text and retain each check's measured
ranges. A flag is a lead for reading the passage.

## Comparisons

The comparison uses the same two installed commercially licensed
writing checks. No model weights changed. The five-sentence English
samples have one labeled rewritten interior sentence. The other four
are unchanged from the paired original. Each source work stays in a
single study role. Only the submitted text reaches the checks. The
first study uses 30 news and 30 abstract works, with equal counts from
two rewrite generators in each genre. The second uses a different
set of works and fits references and cutoffs independently for news
and abstracts. Both studies pin the publisher shards and exclude the
recorded source IDs of the available preceding evaluation panels.
[Benchmark methodology](https://arxiv.org/abs/2606.06481)
· [Existing detector terms and source measurement](writing-missed-passage-transfer-2026-11-14.md)

The current-rule proxy selects classifier or ratio flags and uses the
recorded exact sentence or its broader measured context. Raw
agreement requires both to flag the same center and an exact ratio
reduction. In the calibrated comparison, each authentic reference
work contributes its strongest reading in each of three scored-token
length bands. Each submitted candidate gets a rank for each check
against those work maxima. The rank is the count of strictly lower
reference readings divided by one plus the number of reference works.
The three policies compare the stronger
rank, the smaller rank and their mean. A policy selects a unique top
sentence only when it exceeds a cutoff set on separate ordinary
works. The study records two operating points per rule. A useful
location covers half of the edited sentence and reaches 0.30
intersection over union on non-whitespace characters.
[Length-conditional calibration study](https://arxiv.org/abs/2611.10901)

The original calibration uses 32 ordinary government reports for
the length references and 51 other ordinary government reports for
the cutoffs. The follow-up uses 48 ordinary five-sentence references
and 36 ordinary cutoff sources for each target genre. The original
government reports had three five-sentence excerpts per file.
The follow-up uses one five-sentence excerpt per file throughout.

## Government-to-news and abstracts transfer

| Rule | Ordinary files highlighted | Useful rewrite locations | Rewrites highlighted only elsewhere |
| --- | ---: | ---: | ---: |
| Current-rule proxy | 55/60 | 21/60 | 34/60 |
| Raw agreement at the same center | 8/60 | 4/60 | 4/60 |
| Calibrated stronger rank, either point | 1/60 | 0/60 | 1/60 |
| Calibrated smaller rank or mean, either point | 0/60 | 0/60 | 0/60 |

Raw agreement found three useful news locations and one in abstracts.
All eight ordinary agreement flags occurred in news. The pool of
inputs is short English news and abstracts with known single edits.
These are review-location measurements. They do not estimate the
probability that the highlighted text was AI-generated.

## Same-genre calibration

The follow-up held set contains 26 news and 30 abstract works. The
target was 15 works per generator and genre. Eleven news works from
one generator remained eligible after applying the prior-work and
phrase-overlap exclusions. The selected source works never change
roles. The original government operating points were also applied to
these same held files as a transfer control.

| Rule | Ordinary files highlighted | Useful rewrite locations | Rewrites highlighted only elsewhere |
| --- | ---: | ---: | ---: |
| Current-rule proxy | 52/56 | 29/56 | 25/56 |
| Raw agreement at the same center | 8/56 | 1/56 | 7/56 |
| Mean rank, broader government point | 1/56 | 0/56 | 1/56 |
| Mean rank, tighter genre point | 7/56 | 1/56 | 6/56 |
| Mean rank, broader genre point | 14/56 | 5/56 | 10/56 |
| Smaller rank, tighter genre point | 8/56 | 3/56 | 7/56 |
| Smaller rank, broader genre point | 17/56 | 3/56 | 11/56 |
| Stronger rank, broader genre point | 14/56 | 2/56 | 13/56 |

The mean rank at the broader genre point found one news and four
abstract rewrites. Used as an additional cue alongside the current
rule, it would add one correct abstract location, with zero new
ordinary flags or misplaced locations. Both smaller-rank points
add the same location. That single gain leaves the calibrated rules
outside the app. The results reflect source excerpts with a known
rewrite and separate groups for each study role. They do not
establish accuracy for other genres or generators.
[Government-transfer measurements](evaluation-panels/writing-joint-tail-transfer-2026-11-14.json)
· [Same-genre measurements](evaluation-panels/writing-joint-tail-matched-2026-11-14.json)

## Customer reports

The local Full worker processed one ordinary excerpt and one rewrite
chosen because both checks had flagged a location in the benchmark.
Each used a simulated USD 9.99 Stripe checkout. The jobs started only
after verified payment. The evidence API, JSON and Standard PDF
recorded the different text ranges in both reports. Both signed
packages verified against their exact submitted originals. The test
checks report behavior on known examples.
[Worker receipt](evaluation-panels/writing-joint-tail-worker-2026-11-14.json)

## Local replay

The public receipts preserve source hashes, positions, raw readings and
the control-fit parameters. They omit source prose. From the repository
root, run the numerical replay with standard-library Python.

```sh
export PYTHONPATH=backend
first=docs/evaluation-panels/writing-joint-tail-transfer-2026-11-14.json
second=docs/evaluation-panels/writing-joint-tail-matched-2026-11-14.json
python3 -B -S -m scripts.writing_joint_tail_transfer review "$first"
python3 -B -S -m scripts.writing_joint_tail_transfer review "$second" --previous-transfer "$first"
```

For a source rerun, use the pinned test-split news and abstracts CSV
shards in their genre subdirectories. Run `freeze` with `--shards`
and the first receipt as `--prior`. Run it with `--previous-transfer`
for the second roster. For the follow-up, run `classifier` and `ratio`
for `--role fit` and `--role point` first. Use those four outputs as
the `--parts` for `calibrate`, then measure the held role. The
`analyze` phase consumes the committed measurements and frozen
calibration. The same standard-library reviewer can compare the two
receipts on another machine without installing model weights.

For the example reports, use the original roster from the first
comparison, the first public receipt and an empty local output path.

```sh
backend/.venv/bin/python -B -m scripts.writing_joint_tail_worker "$first_roster" "$first" "$local_run"
```
