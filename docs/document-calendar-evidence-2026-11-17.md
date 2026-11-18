# Calendar date evidence in documents

Chipmunk Verify checks explicit English weekdays paired with a day,
spelled-out month and year from 1900 through 2099. It compares the stated
weekday with the Gregorian calendar and highlights the original phrase
when they disagree or the calendar date is impossible. The analysis view,
PDF and JSON use the same character interval. JSON also carries the PDF
page and source boxes or the original text ranges in a Word paragraph,
email body or plain text file.

The check reads mapped text in plain text, DOCX, PDF text layers and email
bodies. Text without a complete year and quoted replies fall outside this
scope. Each observation carries zero weight in the AI assessment.

## Source-held evaluation

The study read 2,543 agency notice date sections from 56 official 2019
Federal Register issues. Of those, 305 had complete dates within scope,
for 420 checked weekday and date pairs. A separate Gregorian weekday
formula set the expected value. One genuine notice contained a calendar
disagreement. Its text says Tuesday, September 24, 2018. The date fell
on a Monday. [Official issue XML](https://www.govinfo.gov/bulkdata/FR/2019/08/FR-2019-08-09.xml)

Ninety notices were selected by source hash after applying the source
and date grammar, before looking at the application results. Each
provided three altered versions and two matched controls. Each altered
version required an exact phrase range and the correct computed weekday
or invalid-date result. The altered variants are controlled changes to
government text. The study measures calendar verification.

| Delivery | Correct exact locations | Untouched or restyled controls flagged |
| --- | ---: | ---: |
| Replaced weekday | 90/90 | |
| Changed the day of the month | 90/90 | |
| Inserted February 30 | 90/90 | |
| Restyled in day-month-year order | | 0/90 |
| Removed the year | | 0/90 |
| Unaltered date sections | | 1 confirmed disagreement in 420 checked pairs |

The original U.S. government text was obtained from GovInfo. The shared
receipt keeps per-issue hashes, notice IDs, original text commitments,
test ranges and numeric outcomes. [GovInfo Federal Register downloads](https://www.govinfo.gov/bulkdata/FR)
· [GovInfo use policies](https://www.govinfo.gov/about/policies)

## Local verification

Five isolated test-mode checkouts covered plain text, DOCX, an email
with a quoted reply, a PDF and the unaltered control. The actual $9.99
checkout gate was exercised with simulated payment. No analysis jobs
were created before simulated payment confirmation. The exported
PDF and JSON had matching text positions. The signed packages replayed
the calendar check against the submitted originals. The newsroom PDF
retained the count and withheld the original date phrase.

```sh
cd backend
receipt=../docs/evaluation-panels/document-calendar-held-2026-11-17.json.gz
python3 -B -S -m scripts.document_calendar_panel review "$receipt"
.venv/bin/python -B -m scripts.document_calendar_panel measure \
  "$daily_xml" "$fresh_run/held.json.gz" --examples "$fresh_run/examples"
.venv/bin/python -B -m scripts.document_calendar_app "$fresh_run/examples" "$fresh_run/app"
```

The receipt's `issues` list gives the exact download and SHA-256 for
each XML file. [Held receipts](evaluation-panels/document-calendar-held-2026-11-17.json.gz)
· [Checkout and original replay](evaluation-panels/document-calendar-app-2026-11-17.json)
