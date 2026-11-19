# Printed AI-use headings in PDFs

The local evidence view can link to one standalone printed AI-use
heading in a supported PDF. It shows the saved wording, page and
extracted character range. The PDF report prints the same location.
JSON includes the original page coordinates for the line and each word.
The heading carries zero weight in the AI assessment, and the writing
checks run on their existing inputs.

The check covers up to 12 pages and needs one complete, selectable,
horizontal heading on an unrotated page. It confirms the extracted word
boxes against the submitted PDF and checks render contrast within those
boxes. The page must also pass the existing hidden, covered and
substituted-text checks. Duplicate headings, OCR-only text, combined
heading-and-body blocks, and the page margins are outside this scope.
The local setting is `VERITAS_DOCUMENT_DISCLOSURE_HEADING_REVIEW`.
It defaults to false.

## Independent source check

The earlier [20-paper JOSS audit](writing-joss-disclosures-2026-11-18.md)
showed errors when interpreting the scope of a statement and when
including nearby sections. The heading-only check reproduced 19 exact
positions across those papers. They contained 57 word boxes. The
remaining paper had no matching heading. This is a replay of the
same source roster with a narrower measurement.

The check located five of seven controlled headings in quotations,
examples, and disclosure-like layouts. Those included an example
introduced in a separate paragraph and a fabricated first-person
statement. The app presents both as printed text with a source
position. The check withheld headings with explicit quotation marks
around the title and files with repeated headings. Among eight
new layout and visibility controls, only the intended standalone
visible heading passed. Three separate published PDFs exceeded the
12-page scope and were excluded from the supported-format denominator.

An independent reader replayed the line positions and 75 word boxes
in all supported cases using the original bytes. The heading matcher
measures a visible title's location. The truth of any statement and
who wrote a passage require separate evidence.

## Local application verification

Six simulated USD 9.99 checkouts verified the evidence page, PDF,
JSON and signed original-file replay. The printed heading appeared
for the visible heading and the example introduced as copied. It
was absent when the option was disabled, without a matching title,
when printed white-on-white, and when repeated. The same submitted
file produced the same overall assessment with the option enabled
and disabled. No analysis started before payment.
