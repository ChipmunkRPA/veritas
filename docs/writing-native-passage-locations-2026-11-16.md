# Locate separate writing passages in Word, PDFs and emails

The earlier selection compared individual sentences with nearby
context in plain text. This test transfers its frozen calibration to
native Word paragraphs, PDF writing and email replies. The common
inference reads the sentences from each actual uploaded file. The
evidence pairs a selected excerpt with its location in that file.
[Earlier passage measurement](writing-multi-passage-transfer-2026-11-16.md)

## What the check covers

- Word main-body paragraphs with a complete character map. The result
  records the source paragraph, XML body position and character ranges.
- Upright PDFs with fully mapped selectable text. The app and report
  give the page numbers and excerpt. JSON retains the original word
  boxes in PDF page points. The scope requires complete visibility
  checks and excludes scanned, unmapped and obscured pages.
- Plain email or simple, source-mapped HTML email. The input is the
  leading mapped reply body. Quotation and signature markers define
  where that body ends. The record gives offsets in the selected
  decoded MIME part. The last two-sentence context is scored again
  when the next sentence is outside the body.

The development option uses the existing `VERITAS_TEXT_PASSAGE_INTERVALS_REVIEW`
setting and the candidate writing configuration. It inspects 12 to 128
English sentences in Full and Deep analyses. Formatting or revision
structures outside the verified source profiles can still be assessed
by the app's other writing checks. These selected passages carry zero
weight in the overall AI assessment.

## Matched-format transfer

Use the 49 held ordinary and mixed pairs from the previously published
test layout. Generate one native Word file, one plain email reply, one
mapped HTML reply and one selectable PDF for each of the 98 excerpts.
The email variants include earlier quoted text. The plain email also
includes a recognizable signature. The PDF fixture uses an embedded
DejaVu Sans font, ordinary line wrapping and page breaks. It leaves
the source characters in the test text. The test does no fitting.

Reuse existing readings when the exact sentence or context matches.
Score changed inputs locally under the frozen checkpoint and select
ranges from the six-decimal readings used by the app. Compare known
edit positions on a character clock that ignores whitespace. Record
ordinary alerts even if the PDF text differs from its pre-rendered
reference. When that correspondence is unavailable, retain the label
in the overall denominator and withhold a localization match.

| Submitted format | Labeled passages located | Ordinary files flagged | Unassessed uploads | Same selections and coverage as TXT |
| --- | ---: | ---: | ---: | ---: |
| Plain text | 24/98 | 6/49 | 0/98 | 98/98 |
| Word | 24/98 | 6/49 | 0/98 | 98/98 |
| Plain email | 24/98 | 6/49 | 0/98 | 98/98 |
| Mapped HTML email | 24/98 | 6/49 | 0/98 | 98/98 |
| Selectable PDF | 24/98 | 6/49 | 1/98 | 90/98 |

Word and both email formats preserved all sentence and context inputs.
The PDF's page breaks and line wraps changed sentence geometry. Only
21 PDFs produced the exact plain-text classifier inputs. The replay
measured 797 additional inputs with the frozen calibration. Two PDF
files from one source differed in their non-whitespace characters
after rendering and extraction. Both returned no ranges, and the audit
withholds a localization comparison for their text. A third PDF
produced 129 sentences and fell outside coverage. Each carrier produced
four unmatched suggestions on comparable mixed excerpts. These are
matched format deliveries of the same works under one renderer.
[Per-delivery comparison](evaluation-panels/writing-native-passage-locations-2026-11-16.json.gz)

## Local app verification

The same mixed economic report gave two selected excerpts in all four
native upload formats. The Word report showed the first range across
paragraphs 1 through 4 and the second across paragraphs 13 through 18.
The source PDF showed the first on page 1 and the second across pages 2
and 3. The plain and HTML replies preserved their own original offsets
within the selected MIME part. A source-matched ordinary email still
showed no passages when the mixed writing was appended as an explicit
quotation.

The evidence API, JSON and exported report agree on the excerpts and
their positions. Signed verification rereads the source and checks
the stored positions. The web and PDF place the excerpts at the start
of the evidence section. Each printed excerpt stays with its source
locations. Ten simulated USD 9.99 purchases queued analysis after
payment. The mixed email gave the same overall assessment with the
option disabled and enabled. The original plain-text signed package
still replays.
[Local application receipt](evaluation-panels/writing-native-passage-app-2026-11-16.json)

## Reproduce

The portable command recounts selection overlap and ordinary alerts from
the published numeric results. It produced the same output on Python
3.12 and 3.14. The carrier replay re-creates and reads
the actual files from the [earlier source plan and measurements](writing-multi-passage-transfer-2026-11-16.md#replay).
Use the pinned classifier snapshot and the DejaVu Sans font identified by
the receipt's SHA-256. ReportLab creates the private test PDFs. The
released data contain hashes, positions and outcomes.

```sh
export PYTHONPATH=backend
panels=docs/evaluation-panels
module=scripts.writing_native_passage_panel
python3 -B -S -m "$module" review "$panels/writing-native-passage-locations-2026-11-16.json.gz"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
"$runner" -B -m "$module" score "$prior/plan.json.gz" "$prior/measured.json.gz" "$snapshot" "$font" "$run/transfer.json.gz"
"$runner" -B -m scripts.writing_native_passage_app "$prior/plan.json.gz" "$run/transfer.json.gz" "$font" "$run/prepaid"
```
