# Calibrating the writing-pattern ratio by sentence length and context

The existing writing-pattern ratio combines two numbers. One measures
how surprising the actual words were given the preceding text. The
other measures surprise for a reference distribution of possible
continuations. The first divided by the second gives the raw ratio.
Either quantity can vary with text length, subject and writing style.
A low ratio supplies a reading location, and ordinary text can produce
the same response. This study measures how often each rule makes useful
selections. [Recent length-calibration study](https://arxiv.org/abs/2611.10901)

## Comparison

This retrospective study uses four rules on the same measured sentences.
Each candidate needs an exactly scored interior sentence and both
neighboring sentences. The reference is the current absolute ratio.
The two context rules compare the sentence with the median ratio for the
rest of its excerpt or with the lower of its two neighbors. The fourth
rule ranks the candidate against genuine sentences of similar token
length. Its three bins contain 18, 29 and 16 fit works, respectively.
Each work contributes its lowest measured ratio within each bin.
The empirical ranks are internal screening measurements.

Fifteen genuine calibration works with no primary classifier flag
supplied each rule's two operating points. Each rule needs a unique
winner and a score strictly above the highest or second-highest
recorded candidate score. The second operating point allows one
additional genuine calibration flag when scores are distinct. It is
small-sample calibration with no population false-alarm guarantee.
The comparison preserves every classifier flag.

The source measurements come from the prior news, abstract and essay
panels and a separate generator transfer. They use the existing
Apache-2.0 model pair, with no new weights or inference in this step.
The original local absolute-ratio comparisons and their delivery
duplication limits still apply. [Native measurements and source qualifications](writing-missed-passage-transfer-2026-11-14.md)

## Results

Each entry counts added useful locations, added genuine flags, and
new wrong locations in previously unflagged edited files, in that
order. The more permissive point is shown below. The original
classifier covered 22 of 132 edited deliveries and flagged 21 of
66 genuine excerpts on the first panel. The raw ratio plus classifier
covered 62 and flagged 58 genuine excerpts.

| Rule | First held panel | Separate generator transfer |
| --- | ---: | ---: |
| Recalibrated absolute ratio | 2 · 1 · 0 | 0 · 0 · 0 |
| Difference from document median | 2 · 5 · 5 | 0 · 1 · 1 |
| Difference from both neighbors | 6 · 7 · 10 | 1 · 2 · 2 |
| References grouped by token length | 3 · 2 · 0 | 0 · 0 · 0 |

The length-based rule gained three exact locations across both parent
generators with two extra genuine flags and no new misplaced locations.
It flagged no excerpts and recovered no edits on the separate transfer.
The two-neighbor rule transferred one correct location with two new
genuine flags and two misplaced locations. Its stricter point found
four additional locations with three genuine flags in the first panel
and no useful addition in transfer. None of the frozen rules passed
the comparison criteria. The app keeps the existing scoring rules.

## Evidence in the app

The browser and Standard PDF now show both measured terms alongside
the raw ratio. The JSON and API already preserve both terms and their
full-precision values. A local simulated USD 9.99 checkout reached the
Apple GPU worker only after payment. For one selected window, the raw
ratio was **0.939902**, from **3.242630 ÷ 3.449966**. The report shows
the rounded terms alongside the exact selected text. The API and JSON
agreed, the signed report verified against the source, and the
rendered evidence page was inspected locally.
[Prepaid analysis receipt](evaluation-panels/writing-ratio-breakdown-app-2026-11-14.json)

## Replay

The public [numeric comparison](evaluation-panels/writing-ratio-context-review-2026-11-14.json)
includes the length-bin references, fixed points and counts. Recount
all eight comparisons with standard-library Python and the checked-in
prior receipts.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.writing_ratio_context_review review \
  docs/evaluation-panels/writing-ratio-context-review-2026-11-14.json
```
