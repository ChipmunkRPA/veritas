# Spread writing evidence across the document

Chipmunk Verify shows up to six detailed writing passages per analysis.
For mapped plain text, Word and PDF documents with more candidates than
fit, the app now gives preference to source blocks with fewer displayed
cards. It preserves the space set aside for evidence where checks disagree
and the existing priority for supported short-passage or second-check
reviews. Within each selection step, classifier support and source order
break ties. Users can see how many candidate passages and source blocks
the displayed cards represent. The recorded measurements still support
each card at its exact source location. The selection carries zero weight
in the overall assessment.

## Held comparison

The selection rule was chosen using the previously exposed government-report
panel. We then sampled 26 unused essay works in two publisher rewrite
families and scored both deliveries using the already installed writing
checks. Each delivery comprises three disjoint source excerpts with
three, five, and three native sentences. The excerpts are separated by at
least 80 characters in the original text. Exactly one interior sentence
in the center excerpt changes. Both labels and source offsets bind that
edit. The reduced three-excerpt layout was set before either writing
check measured the held essays. [Pinned benchmark and reuse terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9)
· [Earlier government-report panel](writing-other-paragraphs-2026-11-14.md)

| Measure across 52 deliveries | Source order | Block-balanced selection |
| --- | ---: | ---: |
| Source blocks represented in the displayed cards | 120 | 125 |
| Native sentences fully within a displayed measured span | 282 | 288 |
| Displayed disagreements | 182 | 182 |
| Known rewrites usefully located | 17/26 | 17/26 |
| Ordinary deliveries with a highlight | 26/26 | 26/26 |

Fourteen deliveries had more candidates than the six-card limit.
Nine changed which cards were displayed. The same individual 17
rewrites met the source overlap criterion under both policies.
The panel contains 24 Gemini 2.5 Flash pairs and two GPT-5.4 Nano
pairs. The latter did not exceed the six-card limit, so the coverage
gain comes from one rewrite family. This is a display selection study
on matched writing passages. The ordinary flags remain a limitation
of the installed writing checks.

## Verify the customer report

Three isolated local analyses submitted the held material as plain
text, Word and PDF after simulated USD 9.99 checkout. Both installed
writing checks completed. Each analysis showed six of seven candidates
across all three source blocks. The PDF and JSON agreed on the coverage.
Every signed package replayed against its own original file with
the correct source block count. The UI regression checks the same
displayed and total counts and the explanation of disagreement.
[Local customer-flow receipt](evaluation-panels/writing-balanced-cards-worker-2026-11-14.json)

## Reproduce

The [numeric receipt](evaluation-panels/writing-balanced-cards-2026-11-14.json)
contains source hashes, ranges and saved responses. It omits source
prose and model weights. Model-free replay runs both selection policies
and checks that each held work was absent from the pinned prior panels.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.writing_balanced_cards_transfer review \
  docs/evaluation-panels/writing-balanced-cards-2026-11-14.json
```

For a native rerun, use the two pinned essay shards in `SOURCES` and
the backend virtual environment. Run `freeze` with `--shards` and pass
the numeric receipt as `--earlier` to bind the source exclusions. Run
the `classifier` and `ratio` phases of
`scripts.writing_sibling_paragraph_transfer` with `--role held`.
Pass the two measurement files to `analyze` in the balanced-card module.
