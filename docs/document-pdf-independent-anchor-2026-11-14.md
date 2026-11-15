# Compare a changed sentence across PDF pages

In a local Chipmunk Verify document analysis, open **Compare with an earlier
draft** and select **Experimental paired writing check** before uploading the
earlier PDF. When a changed sentence has two unique unchanged neighbors in the
same selectable text block, the comparison shows the changed words and the
page from each source. The page numbers can differ. Where the wording
comparison contains matching changes, the finding links to their highlighted
source crops. The comparison PDF prints both source locations. The JSON also
records the word boxes in each PDF's coordinate system.

The optional check covers short English drafts with visible selectable text,
upright pages in a verified single-column order, and at most 12 pages per
file. Each checked sentence needs a separate unambiguous match. Scans,
unmapped content, hidden text and changes crossing PDF text blocks are
outside this check. The local feature uses
`VERITAS_DOCUMENT_PAIRED_INDEPENDENT_REVIEW=true` with the existing earlier-draft
and paired-writing options. The paired responses carry zero weight in the
overall authorship assessment.

## Local verification

In three simulated prepaid analyses, the changed sentence moved from page
1 in the earlier PDF to page 2 in the revised PDF. The cases used two held
report rewrites from different generators and one ordinary wording change.
The check mapped all three and replayed 26 word boxes. Neither rewritten
sentence showed a joint response increase. This exercise verifies source
locations and the customer workflow. The separate
[single-upload experiment](writing-matched-encoder-2026-11-14.md) records
detector accuracy on a larger held set.

The regression also rejects hidden text for the paired check and records a
failure when a previously exported source page number is altered. Replay a
comparison using its JSON export and both original PDFs.

```sh
cd backend
.venv/bin/python -m scripts.review_document_revision_export comparison.json earlier.pdf later.pdf
```
