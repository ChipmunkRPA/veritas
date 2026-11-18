# Check explicit length conversions in one document

Chipmunk Verify now highlights incompatible length conversions in mapped
plain text, Word, selectable PDF, and supported email bodies. For example,
`12 in = 35.00 cm` identifies the original calculation and shows `30.48 cm`
at the same stated decimal precision. The standard PDF and JSON export
carry the same original text and positions as the browser. The PDF uses
the mapped text block and the JSON also carries the matching original word
boxes. Word and email findings retain their mapped character positions.

The check accepts common metric length units and the international inch,
foot, yard and mile. A qualifying equality occupies a complete mapped block
or appears inside parentheses. Its right-hand side may differ by up to half
the last printed place to allow ordinary rounding. The check excludes
approximation and nominal-dimension wording, ignores OCR-only PDF regions
and email quotes, and avoids foot or mile comparisons in text that mentions
survey feet or nautical miles. Unsupported and differently dimensioned
units stay outside scope. A disagreement can come from ordinary calculation,
rounded measurements, or transcription. It carries zero weight in the
AI assessment. [NIST international and survey unit definitions](https://www.nist.gov/pml/us-surveyfoot/revised-unit-conversion-factors)

## Published factor comparison

The frozen input plan reads the six-by-six standard length table on page 9
of the pinned NIST Handbook 133 Appendix E PDF. It reconstructs each of the
30 off-diagonal published factors from the original table's numeric word
positions. The held text fixtures convert those table cells to explicit
equalities and preserve the printed digits. A second correct version spells
out the result unit and places the equality inside parentheses. A paired
changed version increases the result by two units at its last printed place.
A fourth version marks that changed value as approximate. An independent
rational calculation derives the same length relationships from the inch,
foot, yard, mile and metric definitions. [Pinned NIST reference](https://www.nist.gov/system/files/documents/2025/12/23/2026_NIST_HB133_AppendixE-20251222.pdf)

| Held fixture | Checked | Disagreements at the exact source expression |
| --- | ---: | ---: |
| Printed values | 30/30 | 0/30 |
| Correct restyling | 30/30 | 0/30 |
| Changed values | 30/30 | 30/30 |
| Approximation wording | 0/30 | 0/30 |

The held test verifies arithmetic, scope, and location in statements
reconstructed from one standard table. The existing two-operand
arithmetic replay still locates all 2,973 staged
changes and reports zero disagreements on the original cases. Existing
signed arithmetic-only reports keep their original verification behavior.
[Held receipt](evaluation-panels/document-length-conversions-held-2026-11-18.json) ·
[Earlier arithmetic replay](document-arithmetic-evidence-2026-11-17.md)

## Application verification

Five local synthetic uploads passed simulated USD 9.99 checkout, analysis,
PDF and JSON exports, and signed verification against the source files.
TXT, DOCX, EML and PDF each identified the same discrepancy. An ordinary
TXT control returned one checked conversion and zero disagreements. The
email replay kept the quoted equation outside scope. The newsroom PDF
retained the count and withheld the expressions and measurements. No
analysis was queued before payment confirmation and no real payment was made.

The updated Word report also passed a Chromium check. The browser and PDF
identified the exact paragraph and decimal value, with matching JSON and
signed source-package evidence. The previously issued report and evidence
package still downloaded byte for byte. [Checkout and release receipt](evaluation-panels/document-length-conversions-app-2026-11-18.json)

## Reproduce

Download the pinned NIST source file. Keep the prepared plan locked before
measuring it. The output includes exact source ranges and both checker
digests. Both prepare and measure reproduced the same bytes on a fresh run.

```sh
export PYTHONPATH=backend
runner=backend/.venv/bin/python
"$runner" -B -m scripts.document_length_panel prepare nist-hb133-e.pdf frozen.json
"$runner" -B -m scripts.document_length_panel measure frozen.json held.json
"$runner" -B -m pytest -q backend/tests/test_document_arithmetic.py
```
