# Complementary text contexts improve localization, with larger false highlights

**Advance the baseline-plus-canonical combination as a development candidate.**
On the reused controlled panel it usefully locates 58/128 mixed documents,
versus 45/128 for the current passage check, and retains every previously useful
location. Genuine-document flags increase from 5/128 to 10/128, spanning five
known source groups instead of three. Wrongly highlighted human text also grows
substantially and must remain part of the decision. No serving default,
calibration or customer confidence changes in this study.

The [procedure](text-layout-methods-v101-protocol.md) explicitly acknowledges
that the earlier formatting results were already known. This is a post-result
development comparison, not fresh confirmation or a blind test. Its freeze was
manually published as `b06ec10`, with remote blob
`fab2be2755696f47bb9e1d37cc367a1a5261fdc1` verified before evaluation.

## What was implemented

An independently written prose normalizer collapses whitespace while retaining
all actual non-whitespace Unicode characters, existing token boundaries and
source positions. Every requested layout must produce the same canonical
string and correspondence. That exact input is matched to an already executed
native model response; no approximate cache match or invented response is used.

The four methods are the submitted-layout current check, the canonical-input
current check, a canonical current/interval mask union, and the union of baseline
with both canonical checks. All retain the separately recorded research points.
The small interval head and permitted pretrained models are unchanged. Mask
unions have no combined authorship probability. Different contexts of one
classifier are not independent models.

## Complete original-layout comparison

There are 288 related 160-word compositions from 32 known source/prompt groups:
128 genuine, 128 mixed and 32 wholly generated. They are consumed development
sources, not 288 independent authors. Useful localization means original
non-whitespace character IoU >= .3; these internal measurement criteria are
not public screening thresholds or customer confidence claims.

| Measure | Current baseline | Canonical current | Canonical two-check union | Baseline plus canonical union |
| --- | ---: | ---: | ---: | ---: |
| Useful mixed locations, /128 | 45 | 19 | 36 | 58 |
| Useful wholly generated locations, /32 | 22 | 22 | 30 | 30 |
| Useful 60-word insert locations, /32 | 18 | 2 | 5 | 19 |
| Useful 12-word insert locations, /32 | 0 | 0 | 0 | 0 |
| Genuine documents with a highlight, /128 | 5 | 7 | 9 | 10 |
| Genuine groups with a highlight, /32 | 3 | 4 | 4 | 5 |
| Human characters highlighted in genuine documents | 1,643 | 2,569 | 5,876 | 6,094 |
| Human characters highlighted in mixed documents | 504 | 1,356 | 5,975 | 6,337 |
| AI characters located, /95,802 | 33,647 | 26,080 | 56,202 | 61,525 |
| AI characters missed, /95,802 | 62,155 | 69,722 | 39,600 | 34,277 |
| AI intervals at least half recovered, /256 | 56 | 35 | 98 | 113 |
| AI-containing documents with no AI characters located, /160 | 73 | 100 | 87 | 66 |

The combined candidate gains 13 useful mixed locations: seven mostly-AI
documents with a human gap, one larger AI insert and five multi-region cases.
It also gains eight wholly generated locations. No baseline useful location is
lost on the original-layout panel. Mean mixed character IoU rises from .20089
to .31103. That gain comes with more wrong human highlights, not just five
additional genuine-document alerts. Counting alerts alone would conceal the
larger false-highlight area.

All four methods locate **zero AI characters** in all 32 short-insert cases.
The baseline and combined candidate highlight 220 human characters in that
condition. A broader document response does not close or substantiate the
12-word localization gap.

## Layout stability and coverage

The two canonical-only methods preserve exactly the same input and highlights
across all five requested layouts. That is expected from construction; it is
not independent evidence of authorship accuracy. Canonical current alone loses
27 useful AI-containing locations and gains one relative to baseline, so
normalization alone is not the proposed replacement.

The combined candidate deliberately retains the submitted-layout baseline and
is **not invariant**. Its highlights change in 31/288 documents after the
40-word paragraph intervention, compared with 91/288 baseline; single-space
text changes 32/288 combined versus 58/288 baseline. Stable highlights can still
be wrong. Full tables for all recipes, methods and requested layouts are in the
[aggregate result](evaluation-panels/text-layout-methods-result-v101.json).

All 5,760 method/view/document observations were assessed. The retained current
check provides complete non-whitespace source coverage on this panel, so the
reported methods have no unscored source characters here. This does not prove
coverage on long documents, unsupported Unicode or other deployment inputs.
An unavailable component is unassessed rather than an observed negative.

## Execution and separate arithmetic review

The comparison exited zero in 1.48 seconds, with maximum observed self RSS
162,578,432 bytes. Observations were per document, not continuous peak values.
It executed zero new neural calls and imported no scientific/neural runtime.
It reused 800 distinct previously recorded inputs to analyze 1,440 requested
views; those equivalent views are not independent confirmations.

The [separate reviewer](text-layout-review-v102-protocol.md) was frozen and
manually published as `0210254`, remote blob
`4aab9366178cd09f0689ff49c85a4c091e0724a1` verified before execution. It imports
neither the new scorer nor model/application/scientific modules. A separate
split/join implementation checks source lineage and token boundaries, projects
the original proposals, recomputes every mask statistic and reproduces all
5,760 observations, recipe/aggregate counts, group flags and layout changes
exactly. The [review result](evaluation-panels/text-layout-review-result-v102.json)
passed. It does not independently rerun neural inference, refit calibration,
verify source authorship or constitute outside human review.

The owned-child permission preflight passed before the monitored review launch
outside the sandbox. The review exited zero in 1.85 seconds with no guard or
sampling errors and maximum sampled descendant RSS 164,511,744 bytes. The
[resource profile](evaluation-panels/text-layout-review-profile-v102.json)
retains the exact observed bounds. Both stages kept 900-second/1.5-GiB data-only
limits and RAM/disk reserves; existing neural budgets were unchanged. All
72 relevant tests passed locally, including 31 new authored method/review cases.

## Next qualification

Keep the current method and rollback files. Retain the combined candidate for
its additional coverage rather than rejecting it solely for extra alerts.
It needs separate calibration and new natural articles, emails, mixed documents
and source/generator controls. The current maps and points were not fitted to
canonical inputs or this combination, so these results cannot supply a new
customer authorship confidence. Short-insert detection needs a different or
improved local method; normalization and this union have not solved it.

The source credit remains TXD-22 v1, Md Sadiq Iqbal and Mohammod Abul Kashem,
[DOI 10.17632/prcjcggtjf.1](https://data.mendeley.com/datasets/prcjcggtjf/1),
publisher-declared CC BY 4.0, with the earlier lineage/training-overlap limits.
Source text, pairings and per-case journals remain private; published material
contains code, aggregate counts and whole-artifact hashes. Outside human
reviewers remain zero. No outreach, GitHub Actions, real charge or GCP deployment.
