# Arithmetic evidence in document analyses

Chipmunk Verify recalculates explicit two-operand addition, subtraction and
multiplication in complete mapped text blocks or parenthesized expressions.
Addition and subtraction support signed decimals and matching currency
symbols. It highlights the original expression and shows the displayed and
calculated results. The analysis view and standard PDF use the same extracted
character ranges as the JSON. JSON also includes the exact source range for
plain text, Word and mapped email bodies or the text boxes on the relevant PDF
page. Signed packages can replay the measurement against the original file.

The check accepts exact equality in English-style decimal notation. Its scope
excludes division, implicit totals, quoted email replies, results qualified by
an approximation symbol, and PDF regions assessed through OCR. A disagreement
can come from an ordinary calculation or transcription error and has zero
weight in the overall AI assessment.

## Local evaluation

The 1,319 held GSM8K problems contain 4,282 worked calculation annotations.
Of these, 2,973 met the benchmark's two-operand grammar. An independent exact
rational calculation agrees with every original result in that set. The
check found all 2,973 deliberate changes to the stated values at the correct
text positions. It raised zero review flags on 2,973 correctly restyled
equations and left all 2,973 approximate expressions outside scope. These
are paired arithmetic tests on human-worked material, so the counts measure
calculation and position accuracy.
[Publisher dataset and annotation format](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c)
and [MIT license](https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE).

A separate local checkout test ran five uploads through the application.
TXT, DOCX, EML and PDF all showed the same incorrect amount in the
evidence endpoint, JSON export, standard PDF and replayed signed package.
The fifth upload had a correct calculation and produced zero disagreements.
The email contained a second incorrect calculation in a quoted reply,
which was outside scope. The redacted PDF retained the check counts while
withholding the expression and values. The simulated billing transport
verified that each analysis was scheduled only after payment confirmation.
The application's normal minimum of 15 document words applies.

## Reproduce

The [local evaluation receipt](evaluation-panels/document-arithmetic-held-2026-11-17.json)
pins the test data, license, implementation and recipe by SHA-256. The
[checkout receipt](evaluation-panels/document-arithmetic-app-2026-11-17.json)
records the exported source coordinates. From `backend/`, run the recipe
against the [publisher's exact test file](https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl)
and its license.

```sh
.venv/bin/python -B -m scripts.document_arithmetic_panel test.jsonl LICENSE held.json
.venv/bin/pytest -q tests/test_document_arithmetic.py
```
