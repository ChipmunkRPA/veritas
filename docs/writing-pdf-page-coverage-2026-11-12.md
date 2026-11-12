# Verify writing coverage page by page

New PDF evidence sets record which source pages supplied selectable text, recognized text, both, or no mapped text. The ordinary result, Standard PDF and JSON distinguish characters with a recorded local response, characters read as context, and mapped characters with no local record. Unchecked image areas remain identified even when a page has scored text. Consecutive pages without mapped text appear as a single range. [Local verification receipt](evaluation-panels/writing-pdf-page-coverage-app-2026-11-12.json)

Counts exclude whitespace and use the exact frozen source map. The page totals cover the retained extraction even when the separate passage-range list reaches its 256-range limit. The Standard PDF shows up to 20 page rows and points to the complete JSON table. The page record describes extraction and scoring coverage. It carries no authorship confidence weight.

## Local verification

Two matched four-page PDFs used the same layout and two licensed F1000Research source works. The genuine control used the publishers' dated revised sentences. The other PDF substituted two known rewrites from different offline generators, one in selectable text and one in a scanned page. Both PDFs included a blank page and a page with selectable text over an unchecked image area. A third file combined native text with a blank page and exercised signed source replay without a successful OCR transcription. [Source and generator qualification](writing-source-aligned-transfer-2026-11-12.md)

| Source page | Edited PDF | Matched genuine PDF |
| --- | --- | --- |
| Page 1, selectable text | 471 characters with a recorded local response | 464 characters with a recorded local response |
| Page 2, blank | No mapped text | No mapped text |
| Page 3, selectable text and unchecked image area | 30 characters with a recorded local response | 30 characters with a recorded local response |
| Page 4, recognized text | 248 characters with a recorded local response | 264 characters with a recorded local response |

The two edited passages kept their exact source positions on pages 1 and 4. The worker recorded context responses covering both passages and selected **zero passages** for review in the edited file. The classifier missed both known edits. Coverage makes that distinction visible.

All three isolated Full jobs used simulated checkout and the installed writing classifier. Unpaid events queued zero jobs. An independent calculation rebuilt the page counts from the saved response ranges and original text mapping. The browser, API, JSON and Standard PDF agreed on the page entries. The report table was also rendered and inspected. All three signed packages passed replay with their originals.

## Replay and history

Signed source replay re-extracts the PDF and checks source page numbers, extraction mode, mapped character totals, unchecked areas and exact retained text ranges. It checks saved state counts against the retained ranges and runs no authorship inference. For ranges outside that list, the package preserves the signed state totals. The service verifies those totals against its full frozen response records when preparing or loading the evidence.

~~~sh
python -m app.cli verify evidence.zip --public-key instance-key.pem --source-document original.pdf
~~~

The page table appears in newly built evidence sets. An earlier signed scanned-PDF package without the optional page field still passed original-source replay, and its existing PDF bytes matched the saved report exactly. Existing frozen evidence sets and issued PDFs retain their earlier contents.
