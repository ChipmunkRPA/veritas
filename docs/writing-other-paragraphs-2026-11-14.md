# Reading a writing-pattern flag alongside other passages

The current writing-pattern check can highlight ordinary formal reports.
This experiment asks whether comparisons within the submitted document
can make those highlights more specific. One low ratio can reflect a
local change or a pattern shared by other passages in the same report.
The new selector stays outside the app. It removed fewer ordinary
highlights than useful rewrite locations at both frozen points.

## Study design

Each case uses three disjoint excerpts from one English government
report. The publisher's text flattens its source layout. We present the
three excerpts as separate paragraphs, with five sentences each. At
least 200 characters separate the excerpts in the flattened source.
The revised delivery changes exactly one sentence in the middle
paragraph. The benchmark's inline labels and character offsets establish
the edit, and the other 14 sentences match the ordinary delivery.
Source IDs and normalized 12-word overlap are checked against earlier
study rosters and across admitted works.
[Pinned dataset and reuse terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9)
· [Benchmark methodology](https://arxiv.org/abs/2606.06481)

An eligible writing-pattern flag covers exactly one measured interior
sentence. Each of the other two paragraphs supplies interior
measurements with similar scored-token counts. The selector compares
the flagged ratio with each paragraph's median and takes the smaller
difference. This makes the stronger comparison depend on both
reference passages. The test retains all existing passage-classifier
flags and tests two internal ratio operating points. An eligibility-only
comparison keeps every measurable ratio flag, regardless of the contrast,
to separate the effect of the new rule from the coverage restriction.
Separate ordinary development works set both points before held
inference. The points have no customer confidence meaning.
[Exact ratio inputs and source ranges](writing-missed-passage-transfer-2026-11-14.md)
· [Sentence-level segmentation research](https://arxiv.org/abs/2509.17830)

## Held results

Of 32 ordinary calibration reports, 29 had at least one eligible
comparison. The tighter point retained two of their highest contrasts.
The looser point retained seven. The held set used 51 different source
works with 40 Gemini 2.5 Flash and 11 GPT-5.4 Nano rewrites. Each
work had a matching ordinary delivery.

| Rule | Ordinary files highlighted | Useful rewrite locations | Rewritten files highlighted only away from the edit |
| --- | ---: | ---: | ---: |
| Passage classifier only | 39/51 | 13/51 | 26/51 |
| Existing classifier and writing-pattern selection | 51/51 | 29/51 | 22/51 |
| Classifier plus ratio eligibility only | 50/51 | 21/51 | 28/51 |
| Classifier plus tighter paragraph contrast | 43/51 | 13/51 | 29/51 |
| Classifier plus looser paragraph contrast | 45/51 | 15/51 | 28/51 |

A useful location covers half the changed sentence and reaches at least
0.30 intersection over union on non-whitespace characters. The table
includes cases with no measurable contrast. Only 47 ordinary and 43
rewritten deliveries supplied at least one eligible comparison.
The eligibility restriction alone lost eight useful locations and
removed one ordinary highlight. The tighter rule lost another eight
locations. The remaining ordinary flags give no reason to promote the
new selector to a default. These measurements count all qualifying
windows. Customer reports cap how many windows they display.

## Customer report sample

The local app ran eight simulated USD 9.99 checkouts on the first two
held source works per generator, using each ordinary and rewritten
delivery. Both installed writing checks completed on every upload.
The app displayed at least one highlight on all four ordinary files
and usefully located three of four rewrites. All eight overall
assessments were inconclusive. No analysis ran before simulated payment.
The evidence API and JSON export agreed for each file. One Standard PDF
and signed package per generator verified against the submitted original.
[Numeric worker receipt](evaluation-panels/writing-other-paragraphs-worker-2026-11-14.json)

## Replay

Recount the fixed measurements and the calibration from the
[numeric receipt](evaluation-panels/writing-other-paragraphs-2026-11-14.json)
using only Python's standard library from the repository root.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.writing_sibling_paragraph_transfer review \
  docs/evaluation-panels/writing-other-paragraphs-2026-11-14.json \
  --worker-receipt docs/evaluation-panels/writing-other-paragraphs-worker-2026-11-14.json
```

The receipt has labeled offsets, source hashes, ordinary calibrations
and numeric responses. It omits report text and model weights. For a
native rerun, place the three CSV files listed in `SOURCES` under
directories for held and calibration shards, retaining the filename
format `{split}-reports_{generator}.csv`. Use the backend's virtual
environment and offline model cache. The `freeze` phase selects sources.
Pass the numeric receipt as `--earlier-panels` to reproduce the pinned
set of earlier source exclusions.
Run `classifier` and `ratio` with `--role calibration`, then `calibrate`
with those two outputs. Run both inference phases with `--role held`,
and call `analyze` with all four response files and the frozen points.
The ratio run used Apple Metal and the classifier and ratio checkpoints
already installed for commercial use.
To repeat the simulated checkout, run `worker` with the same roster and
the numeric receipt as `--benchmark`. Use a new output directory.
