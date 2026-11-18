# Passage evidence in OpenDocument and Rich Text

The optional local writing passage check also supports mapped OpenDocument
Text (`.odt`) and Rich Text (`.rtf`). Positions identify the decoded body
paragraph or text block and its Unicode character ranges. The report uses
the file’s recorded visibility rules and leaves out supported hidden content.
Files need complete mapped body text and 12–128 English sentences under the
existing Full or Deep analysis settings. The check contributes zero weight
to the overall AI assessment. Earlier signed reports retain their original
extraction and recorded coverage. [Document formats and visibility](document-formats-2026-11-17.md)

In local development, use `VERITAS_TEXT_PASSAGE_INTERVALS_REVIEW=true`
with the candidate English writing path. Add
`VERITAS_TEXT_PASSAGE_DOCUMENT_BASELINE_REVIEW=true` to consider writing
patterns elsewhere in the same file. The earlier supported formats use
the same existing settings.

## Native format transfer

The transfer fixed the existing pooled and within-document selection with
economic report confirmation enabled. Each source was rendered to both
formats twice, once with a directly hidden paragraph or run and once without
it. The replay required the same exact extracted writing, sentence geometry,
and every recorded context input hash before reusing the earlier model
readings. It checked every selected range against the independently retained
text of its exact decoded blocks. The four carrier variants preserved all
116 input cases and their selections. This comparison measures format
transfer on previously evaluated prose. It supplies no fresh benchmark of
authorship accuracy. [Earlier historical and economic panel](writing-local-baseline-transfer-2026-11-16.md)

| Source group | Useful rewrite locations | Ordinary files with a selected passage |
| --- | ---: | ---: |
| 40 matched historical and economic pairs | 19 of 55 | 6 of 40 |
| 14 independently dated handbook revisions and their earlier files | — | 1 of 14 earlier, 1 of 14 later |
| Eight separate short handbook rewrites | 0 of 8 | — |

The historical alerts came from the same work before and after editing.
Neither precisely matched the changed sentence. One earlier source group
was excluded because its dated revision and a later generated test file
contained identical text. On the 40-pair panel five highlighted ranges
missed any useful marked edit in files with known rewrites. One highlight
missed its known edit in the short-rewrite panel. These limits carry over to
all four native deliveries. [Dated editorial controls](writing-turing-editorial-history-2026-11-18.md) · [Origin overlap audit](writing-remote-semantic-transfer-2026-11-18.md)

## Local checkout and reports

Six simulated USD 9.99 purchases exercised both carriers with the local
writing checks. Zero jobs started before verified payment. A matched
ordinary file in each format had no selected passage. The edited files
selected the same passage across two decoded source blocks. Chromium,
Standard PDF, exported JSON and signed original-file replay agreed on all
four positions. Hidden text remained outside the evidence. Two additional
analyses used the same edited files with the option disabled and preserved
both overall assessments. [Checkout and report receipt](evaluation-panels/writing-decoded-carrier-app-2026-11-18.json)

## Recount

The portable receipt contains numeric ranges, file and input commitments,
and anonymous groups. The review command recounts overlap and coverage and
checks the range structure without installing models. Replaying source
characters independently requires the separately retained writer inputs.
[Numeric transfer receipt](evaluation-panels/writing-decoded-carrier-transfer-2026-11-18.json.gz)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_writing_decoded_carrier \
  docs/evaluation-panels/writing-decoded-carrier-transfer-2026-11-18.json.gz
```
