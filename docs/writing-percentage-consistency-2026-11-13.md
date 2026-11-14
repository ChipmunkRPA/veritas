# Repeated percentage statements in one document

The local TXT check highlights both values when two complete sentences in
separate paragraphs use identical wording, punctuation and percentage units
apart from a single number. It ignores letter case and spacing, and treats
`47` and `47.0` as the same number. Differences in a stated reporting year,
cohort, denominator or citation remain part of the comparison. The check
omits marked quotes, ranges and approximate numbers in its supported syntax.

Enable `VERITAS_DOCUMENT_PERCENTAGE_CONSISTENCY=true` in a local development
instance and run a new TXT analysis. The evidence page, PDF and JSON retain
both complete sentences and the exact Unicode ranges for the values in the
extracted text. Redacted PDFs show ranges and omit the excerpts and values.
This evidence carries zero AI-authorship weight. Two sentences can use the
same wording while referring to different unstated dates, groups or
denominators. Source context is needed to interpret the difference.
The check reads the stored sentence boundaries from the existing document
decode. Loading the analysis and generating a report reuse those boundaries.

## Local validation

The frozen selection used a pinned GovReport pack and salted source order.
Twenty-four previously unused GAO test highlights with eligible percentage
sentences supplied separately authored in-document comparison cases. Each
known edit changes a percentage in a copy of an otherwise identical
published sentence, then appends it as a new paragraph. A second variant
uses a different decimal format. The controls retain the same public
source prose and vary only the defined condition. [GovReport source and terms](https://gov-report-data.github.io/)

| Cohort | Files | Exact disagreements reported |
| --- | ---: | ---: |
| GAO highlights with a planted value disagreement | 48 | 48 |
| Same-source wording and context controls | 288 | 0 |
| Additional untouched GAO test highlights | 128 | 0 |
| Dated or constructed ordinary revisions and originals | 192 | 0 |
| Known AI-edited texts from earlier evaluation panels | 255 | 0 |

The same-source controls include exact repetition, numerically equivalent
forms, different years, cohorts, denominators, units and citations, marked
quotations, approximate values, ranges, repetitions within the same
paragraph and the untouched highlight. The independent untreated GAO sample
had only nine reports with an eligible percentage sentence. The retained
ordinary panel includes historical versions, such as both versions from
21 F1000Research works, plus genuine forum and USWDS versions, eLife originals
and other ordinary controls. A file without a comparable repeated statement
receives no assessment on this check. These results support the narrow
literal comparison. They provide no measured gain on generative edits, and
no estimate of accuracy on semantic paraphrases or unstated context.

The evidence design follows the distinction between finding the two internal
references and deciding whether a discrepancy is meaningful. The FIND
benchmark informed the evaluation design; its data carry noncommercial
research terms and were not used. [In-document inconsistency research](https://arxiv.org/abs/2512.18601) · [FIND data terms](https://huggingface.co/datasets/kensho/FIND/blob/main/README.md)

The [scalar receipt](evaluation-panels/writing-percentage-consistency-2026-11-13.json)
contains file commitments, case roles and positions. The native runner
rebuilds the receipts from the same local GovReport pack and the retained
writing case files. Its source hashes guard both the publisher pack and all
case rosters.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.writing_percentage_consistency_transfer \
  "$gao_pack" "$scratch/writing-percentage-consistency.json" \
  --short-a "$word_edits/roster.json" --short-b "$surprisal/roster.json" \
  --long "$interval/roster.json" --f1000 "$f1000/cases.json" --elife "$elife/cases.json"
```
