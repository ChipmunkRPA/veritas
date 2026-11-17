# Confirming writing highlights in economic reports

Similar writing patterns occur in untouched economic reports. The local
review can now compare a proposed highlight with a separate economic-report
reference before showing that highlight. It keeps the original passage
positions when both comparisons support the same area. The reference applies
only when the file's sentence lengths and score distribution closely fit
the economic reports in the development data. Files outside that reference
continue through the general passage review.

## Separate reference for report-style writing

Both comparisons use the same individual-sentence and surrounding-text
readings. Use the combined document profile and tighter report operating
point from the earlier study. For the report comparison, adjust each
reading for its sentence width using the measured ordinary report
distribution. Retain a highlighted range when the report comparison
selects an overlapping range covering at least half the shorter range.
The study also measures a report-only alternative. The primary rule and
its parameters were frozen before scoring the new panel. [Earlier
calibration and out-of-genre findings](writing-profile-calibration-2026-11-16.md)

Separate measurement by domain and generator follows research on how
writing detection changes across model generations and stages of revision.
The app uses its existing Apache-2.0 writing checkpoint. [Writing domain
and model transfer study](https://arxiv.org/html/2610.11599v1) ·
[OpAI-Bench methodology](https://arxiv.org/html/2606.06481v1) ·
[Pinned detector terms](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615)

## Fresh local measurements

The Board panel uses 24 previously unused Beige Book release dates from
2014–2016, in rotation across Boston, Chicago, and San Francisco. Each
original supplies a fixed two-sentence target within one paragraph.
Two local Apache-2.0 generators each attempted a rewrite per report. A
response had to contain two complete sentences, fit the length rule,
and use substantially different wording. Eight and two responses met
those rules, respectively. All kept the exact original text outside
the labeled edit. [2014 Board index](https://www.federalreserve.gov/monetarypolicy/beigebook2014.htm) ·
[2015 Board index](https://www.federalreserve.gov/monetarypolicy/beigebook2015.htm) ·
[2016 Board index](https://www.federalreserve.gov/monetarypolicy/beigebook2016.htm) ·
[Board source terms](https://www.federalreserve.gov/disclaimer.htm) ·
[Generator weight terms](text-elife-rewrites-v273-method.md)

The separate GAO panel pairs 22 untouched reports with publisher-annotated
polished versions. The originals match the pinned GovReport highlights
and predate 2021. Six pairs come from the OpAI-Bench test split and 16
from its training split. All 22 served as held inputs for this project.
The intake checked prior public receipts for their report identifiers and
text digests and checked which original sentences stayed identical.
Each newer report has two annotated locations. [OpAI-Bench terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench#licensing-and-reuse) ·
[GovReport terms](https://gov-report-data.github.io/#license)

The primary counts use the published layout. The whitespace comparison
preserves the same non-whitespace characters. A correct location needs
30% intersection over union and half of the labeled characters.

| Source | Untouched files flagged, current → report confirmation | Labeled locations found, current → report confirmation |
| --- | ---: | ---: |
| Federal Reserve district reports | 16 → 14 of 24 | 0 → 0 of 10 |
| GAO report highlights | 1 → 1 of 22 | 0 → 0 of 44 |

Seven of the 24 untouched district reports qualified for the specialized
reference. All 22 GAO pairs stayed on the general check. The confirmation
also removed two misplaced highlights in the edited Board files.
The report-only alternative gave the same aggregate counts on these
samples. Both methods missed every new short rewrite. Replaying the
earlier economic-report comparison preserves the six correctly located
rewrites there and clears five of its 14 false highlights. Those earlier
files helped select the method and are separate from the fresh counts.
[Complete numeric and source receipt](evaluation-panels/writing-report-confirmation-2026-11-16.json.gz)

## Local review and reproduction

Set `VERITAS_TEXT_PASSAGE_REPORT_REVIEW=true` with both writing passage
review settings enabled in development. The results view and Standard
PDF explain when the report reference applied. JSON and the signed
package preserve the same ranges. This evidence has zero weight in the
overall assessment. The setting defaults off in the repository.

Six simulated USD 9.99 purchases exercised the local worker. Zero analysis
jobs queued before a paid event. In plain text and PDF, the check cleared
the same selected false highlight on an untouched report. The Word fixture
kept the earlier rewrite range and native paragraph positions. A separate
GAO example used the general passage check. The browser checked all three
report-style views. The PDF, JSON, and six signed packages agreed with
the recorded ranges. The two on/off comparisons kept the same overall
assessment. [Local app replay](evaluation-panels/writing-report-confirmation-app-2026-11-16.json)

The portable verifier replayed 156 delivered versions across 46 source
groups and recounted 36 comparison strata with the separate event matcher.
The native readback checked 5,024 context windows and 140 proposed
ranges against the exact source text. The public receipt contains
numeric readings, digests, declared labels and locations.

```sh
export PYTHONPATH=backend
runner=backend/.venv/bin/python
receipt=docs/evaluation-panels/writing-report-confirmation-2026-11-16.json.gz
python3 -B -S -m scripts.writing_report_calibration verify "$receipt"
python3 -B -S -m scripts.writing_report_calibration verify "$receipt" \
  --plans "$run/board-plan.json.gz" "$run/gao-expanded-plan.json.gz"
```

Reconstruct private source plans using the published source URLs and
digests in the receipt and the pinned [GovReport highlights pack](writing-gao-archive-comparison-2026-11-13.md).
The `record`, `generate`, `assemble`, and `gao` commands in
`scripts.writing_report_sources` give the exact selection and generation
recipe. Keep the native plans and generated text in the local test folder.
