# Read tracked Word paragraph breaks in the accepted text

When Word tracks deletion of a paragraph mark, accepting it joins the two
paragraphs without inserting a space. Accepting an inserted mark keeps the
break. Chipmunk Verify now reads that accepted text for supported Word
body paragraphs. The original DOCX stays intact. Previously both kinds of
paragraph marks could cause the writing checks to be skipped.
[Microsoft Open XML paragraph deletion rules](https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.wordprocessing.deleted#remarks)
· [Microsoft Office paragraph insertion rules](https://learn.microsoft.com/en-us/openspecs/office_standards/ms-oi29500/1d66ba8a-1eeb-4cff-ab8a-eaa79113cc89)

The accepted text can map to more than one original paragraph. The
extractor retains each native body path, so a highlighted word split
by a deleted break still points to both original text ranges. The
browser, PDF and JSON show the revision type and the native XML path.
The separate repeated-percentage check skips the joined paragraph and
continues checking other eligible paragraphs. It requires a complete
sentence within one native paragraph.

The projection covers simple neighboring body paragraphs. A deleted
break next to a table, unsupported moves and complex revision
structures still leave writing unassessed. These are document revision
records with zero AI-authorship weight. Historical reports replay the
extraction policy recorded in their signed package.

## Local results

Both writing checks completed on the Mac GPU. The tracked deletion
and its separately written accepted copy had identical inputs and
scores across 18 windows. The tracked insertion and its counterpart
matched across 20 windows. All four overall results were inconclusive.
The tracked deletion mapped the reconstructed word to two native XML
body paths. The table-boundary control stayed unassessed and produced
no scored window.

Five simulated USD 9.99 checkouts started zero jobs before payment.
The signed packages replayed against all five original DOCX files.
The original-file verifier also replayed two previously issued inline
and unsupported-paragraph reports using their historical policy.
[Local verification receipt](evaluation-panels/document-word-paragraph-breaks-2026-11-15.json)

## Reproduce locally

The local fixture writes the expected accepted document separately
for each tracked edit. The deletion leaves a word split between the
two original paragraphs. The insertion preserves the break between
two blocks of text. A separate fixture puts a table after a marked
paragraph. The test checks simulated prepaid checkout, actual scoring
inputs and responses, PDF output, evidence API and JSON parity, and
signed verification from the original DOCX. The receipt contains
only hashes, positions, coverage and outcomes.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
backend/.venv/bin/python -B -m scripts.document_word_breaks_app "$run/word-breaks"
python3 -B -S -m scripts.document_word_breaks_app \
  docs/evaluation-panels/document-word-paragraph-breaks-2026-11-15.json --review
```
