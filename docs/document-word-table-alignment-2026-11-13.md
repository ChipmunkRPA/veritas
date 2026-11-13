# Find table changes after Word rows or columns move

The earlier-draft comparison can now follow matching rows and columns when
simple Word tables are reorganized. Column headings and row labels must
match exactly and occur once on each side. When a table changes its document
position, both Word files need the same unique table caption. The browser,
PDF and JSON show the earlier and analyzed cell positions separately. They
also show how many rows and columns could not be compared. Repeated, changed,
added or deleted labels remain unchecked. The comparison reports wording
and value changes from two drafts and assigns no AI authorship confidence.

Only top-level tables with the existing simple layout are eligible. The
check keeps the existing limits on cells, tokens and text. Headers, footers,
merged cells, nested tables and tracked markup remain outside this table
comparison. Old paragraph-only and positional table exports continue to
replay under their original comparison schemas.

## Test on CBO release vintages

The panel uses six vintage pairs from five U.S. Congressional Budget Office
datasets. The source files and license are pinned to one repository revision.
We took eight shared date, variable and estimate-type keys per pair by hash
order after excluding keys with conflicting source rows. One pair contained
no revisions among the retained shared cells. The initial intake let one
conflicting CBO key through. The final source audit detected it and rebuilt
the key freeze before the published replay. Selection used neither model
scores nor the size or direction of changes.
[Publisher data](https://github.com/US-CBO/cbo-data/tree/284a95665f9f2f74ed1f482feb629b43fce323da) · [Public-domain license and worldwide waiver](https://github.com/US-CBO/cbo-data/blob/284a95665f9f2f74ed1f482feb629b43fce323da/LICENSE.md)

The Word layouts were constructed from the retained publisher values. The
later versions move the table, reverse and replace rows, and swap two data
columns. The legacy comparison returned no table changes. The new comparison
located all 31 revised cells across five pairs, with 95 verified token
ranges. Six reordering-only controls yielded zero changes. Six controls
added a conflicting value under a duplicate row label. The comparator
reported the ambiguity and returned no change in the affected rows. A PDF
sample and the authenticated-route test retained both source coordinates.
The test also rejects changed export coordinates and checks historical replay.
[Source roster](evaluation-panels/document-word-table-alignment-cbo-roster-2026-11-13.json) · [Results and replay hashes](evaluation-panels/document-word-table-alignment-cbo-2026-11-13.json)

## Reproduce

The public arithmetic reader uses the roster and ledger. For native
replay, obtain `LICENSE.md` and the 12 CSVs under their pinned repository
paths and keep the original directory structure. Change `review` to `score`,
pass a new result path, and provide `--source "$publisher" --output "$drafts"`.
The native runner rebuilds Word drafts, opens exact paired exports, and
checks source ranges and a PDF. These tests use no models.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.replay_document_word_table_alignment review \
  docs/evaluation-panels/document-word-table-alignment-cbo-roster-2026-11-13.json \
  docs/evaluation-panels/document-word-table-alignment-cbo-2026-11-13.json
```
