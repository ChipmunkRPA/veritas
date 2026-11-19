# Locating AI-use disclosure statements in published PDFs

An automated rule matched the declared scope in 11 of 20 published PDFs.
A simpler check located and replayed 19 disclosure headings. Eighteen of
those regions stopped at the correct section boundary. Quoted and nested
headings still caused errors. Both methods stay in research.

## Source and measurement

The audit used 20 papers from the 50 published in the Journal of Open Source
Software's October 2026 issue. The roster was fixed by DOI hash order after
excluding earlier examples. Each file displayed the expected DOI and
CC BY 4.0 notice. A separate reading documented the declared scope and
section boundaries before comparison with the automated result. The test
checked how the statements were located and interpreted. Verifying an author's
statement or the origin of individual prose requires separate evidence.
[Issue 126](https://joss.theoj.org/toc/issue/126) ·
[Publisher licensing](https://joss.theoj.org/about#content-licensing-open-access) ·
[Publisher disclosure policy](https://joss.readthedocs.io/en/latest/policies.html#ai-usage-policy)

| Scope from reading the source | Papers | Automated matches |
| --- | ---: | ---: |
| Manuscript assistance | 13 | 7 |
| Software or documentation assistance, with manuscript use explicitly excluded | 3 | 1 |
| Scope unclear | 3 | 2 |
| Disclosure absent | 1 | 1 |

The rule incorrectly inferred manuscript assistance twice and inferred
global non-use once from an explicit manuscript-assistance statement. A
separate 15-case set of authored controls produced 13 matching results.
An example introduced by a separate sentence and an inverted negation
caused the two failures. An additional authored document with an invented
first-person disclosure produced the same classification as a real statement.
All 13 semantic anchors from 11 papers replayed at their exact source intervals.

The neutral locator replayed the source lines for all 19 headings, including
one correctly bounded disclosure spanning two pages. One region ran into a
separate software-reference subsection. Exact repeated and explicitly
quoted headings caused abstentions. A nested quoted subsection was included,
and a quote introduced in a preceding paragraph caused an incorrect route.
These results support further testing of exact heading navigation on other
publishers and formats. Both checks carry zero weight in AI assessments.

## Recount the published results

The numeric receipt contains anonymous outcome codes, page numbers, counts
and commitments to closed local audits. Publisher IDs, PDF files, and prose
excerpts stay private. The program checks the published receipt and
recalculates the confusion table and source-replay counts. Original
page geometry and extraction were checked separately on the local corpus.
[Numeric receipt](evaluation-panels/writing-joss-disclosures-2026-11-18.json)

```sh
python3 -I -S backend/scripts/review_writing_joss_disclosures.py \
  docs/evaluation-panels/writing-joss-disclosures-2026-11-18.json
```
