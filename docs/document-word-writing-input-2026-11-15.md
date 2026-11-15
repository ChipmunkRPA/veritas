# Use visible Word revision text for writing analysis

The [paragraph-break update](document-word-paragraph-breaks-2026-11-15.md)
extends the original inline projection described here.

A Word document can contain an insertion that is visible in its final
text while the underlying paragraph reader skips the inserted run.
Chipmunk Verify used to show this insertion in the revision evidence
while its writing detectors read the earlier, incomplete extraction.

New DOCX analyses project supported inline revisions before running
writing checks. The accepted view includes the inserted text and
excludes the deleted text. The upload bytes remain intact. The app
records original XML coordinates before projecting the text and uses
the resulting paragraphs when making source maps. This also lets the
existing hidden-text check see directly hidden formatting on an inserted
run. [Word inserted runs](https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.wordprocessing.insertedrun) ·
[Word paragraph mark and revision semantics](https://learn.microsoft.com/en-us/office/open-xml/word/how-to-accept-all-revisions-in-a-word-processing-document)

## Scope and source evidence

The projection covers simple insertions and deletions stored as direct
children of top-level body paragraphs. Paragraph-mark revisions, moves,
table revisions and nested revision structures can change the visible
text in ways this projection cannot resolve. In those cases the app
keeps linked revision evidence and skips the writing assessment. It
also keeps the evidence when applying a supported change leaves too
little readable text.

Evidence shows the original Word XML positions and the affected range
in the accepted paragraph in the browser, PDF and JSON. The accepted
range counts zero-based Unicode code points before whitespace
normalization. The date and author name stored by Word are outside this
evidence. Revision markup describes an edit and adds zero weight to the
AI-authorship estimate. The previous evidence format remains readable,
and offline verification re-extracts the source using the policy
recorded in each package.
[Existing markup evidence](document-word-track-changes-2026-11-13.md)

## Local verification

The native replay tests a tracked file, the same document with changes
already accepted, a file with a tracked paragraph mark, and a
deletion-only file with too little accepted text. It compares accepted
and tracked writing inputs and measured window scores. Both writing
checks ran on the Mac's GPU. The matched versions produced identical
inputs and scores across 24 windows. The paragraph-mark and short files
recorded zero scored windows and returned unassessed. The evidence API
and JSON matched, and two signed Standard reports verified against their
original DOCX files. All four simulated USD 9.99 checkouts confirmed
payment before creating an analysis job.
[Local verification receipt](evaluation-panels/document-word-writing-input-2026-11-15.json)

```sh
export PYTHONPATH=backend
module=scripts.document_word_projection_app
python3 -B -S -m "$module" --review docs/evaluation-panels/document-word-writing-input-2026-11-15.json
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
backend/.venv/bin/python -B -m "$module" "$run/word-projection"
```
