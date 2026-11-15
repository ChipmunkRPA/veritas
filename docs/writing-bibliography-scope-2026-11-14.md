# Writing analysis with a clearly structured reference list

Chipmunk Verify can keep a terminal bibliography out of the writing checks.
It retains the original body text and source coordinates. The browser,
Standard PDF and JSON show the omitted entry count and the section location.
The overall result is limited to body text, with passage evidence retained
and whole-document confidence withheld. The signed evidence can replay the
exclusion against the submitted original.

Set `VERITAS_TEXT_BIBLIOGRAPHY_SCOPE_REVIEW=true` in development to use the
trial. A qualifying TXT, DOCX or native PDF has a standalone References,
Bibliography or Works Cited heading at the end of the body, followed only
by 4–96 complete citation entries. The check looks for author initials,
a year and a DOI or journal volume and pages. It keeps ambiguous lists,
interleaved prose and appendices in the normal input. OCR and incomplete
PDF extraction are outside the trial. Citation accuracy is outside its
scope. [JATS reference-list specification](https://jats.nlm.nih.gov/archiving/tag-library/1.4/element/ref-list.html)

## Layout and local application tests

The source articles came from eLife, Gates Open Research and Wellcome Open
Research. The preparation reads each article-level CC BY 4.0 declaration
and commits the DOI and XML hash. Each case uses two genuine article
paragraphs and reflows six source references into a consistent citation
style. This tests generated layouts with genuine source content. Raw
publisher PDF layouts require a separate test.
[eLife terms](https://elifesciences.org/terms)
· [Gates and Wellcome source preparation](writing-journal-body-transfer-2026-11-11.md)

The initial rule excluded 10 of 12 bibliographies per format and retained
all 180 control layouts. It missed two lists with a multiword surname.
After allowing up to four surname words before a comma, the held test used
12 separate source articles. It excluded all 36 bibliographies across the
three formats and preserved all 180 control layouts. The body text and
native source map matched the body-only comparison exactly in every
exclusion. Both layout receipts reproduced byte for byte.
[Initial receipt](evaluation-panels/writing-bibliography-initial-2026-11-14.json)
· [Held receipt](evaluation-panels/writing-bibliography-holdout-2026-11-14.json)

Fifteen simulated prepaid jobs compared the original documents with the
scope check off and on, and with the body submitted by itself. On the five
documents, 22 of 30 highlighted spans were in reference lists with the
check off. With it on, all 22 reference highlights disappeared and the eight
body highlights stayed at their exact positions. Two matched rewrites
from the same article were still highlighted at the exact edited sentences.
Both writing checks ran in each comparison. Their responses and evidence
matched the corresponding body-only upload. Three native formats passed
Standard PDF, JSON and signed original-document verification. Both checks
still highlighted ordinary body text in the three original excerpts.
[Local worker receipt](evaluation-panels/writing-bibliography-worker-2026-11-14.json)

## Replay

Prepare the publisher XML from the [eLife replay](writing-elife-body-replication-2026-11-11.md)
and [Gates and Wellcome replay](writing-journal-body-transfer-2026-11-11.md).
Use the retained `scored/cases.json` from the second study for the two
worker rewrites. The receipts include numeric measurements and source
hashes. Publisher text and media stay outside the receipts.

```sh
export PYTHONPATH=backend
initial=docs/evaluation-panels/writing-bibliography-initial-2026-11-14.json
held=docs/evaluation-panels/writing-bibliography-holdout-2026-11-14.json
python3 -B -S -m scripts.writing_bibliography_scope review "$initial"
python3 -B -S -m scripts.writing_bibliography_scope review "$held" --previous "$initial"
backend/.venv/bin/python -B -m scripts.writing_bibliography_scope measure "$run/initial" --initial-rule
cmp "$initial" "$run/initial/receipt.json"
backend/.venv/bin/python -B -m scripts.writing_bibliography_scope measure "$run/held" --previous "$initial"
cmp "$held" "$run/held/receipt.json"
backend/.venv/bin/python -B -m scripts.writing_bibliography_scope_app "$run/worker"
```

Use `--elife` and `--journals` for other archive locations and `--revisions`
for the worker rewrite fixture. The worker replay uses the locally
installed detection weights and Apple MPS.
