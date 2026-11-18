# Compare saved HTML articles with an earlier version

Chipmunk Verify can compare an analyzed HTML file with an earlier saved
`.html` or `.htm` copy. Each file must be at most 1 MiB and contain a
single supported article or main area. The comparison reads paragraphs,
headings, lists and descriptions in that area. It reports the number of
compared blocks for each file and identifies when additional text outside
supported blocks was skipped. HTML navigation, figures, tables, linked
resources and scripts remain outside the comparison.

The app shows changed words and punctuation in both versions. Each changed
token has a range in the extracted article and one or more ranges in the
decoded HTML file. A token can span separated ranges when a tag divides the
word. The browser, PDF and JSON preserve the same positions. A reviewer can
compare the exported positions with both original files offline.

An earlier version supplies direct evidence of the wording difference.
The result carries zero weight in the AI assessment. Authorship and version
dates require separate evidence.

## Local comparisons

The paired test reused the 43 previously assessed English excerpts from
12 NASA and NOAA sources. It placed each version in HTML, changed the
navigation and styling, and compared the texts with the authentic article
excerpt supplied separately. Both versions in each pair kept the prepared
prose except for the previously labeled edit or sentence swap. Every reported
changed token was checked against its decoded HTML ranges.
[Original writing study and reuse terms](writing-alignment-preference-2026-11-18.md)

| Case | Pairs | Pairs with displayed wording changes | Rewrite cases with a changed token outside the edited sentence |
| --- | ---: | ---: | ---: |
| First generator rewrites | 8 | 8 | 0 |
| Second generator rewrites | 11 | 11 | 0 |
| Authentic prose with markup changes only | 12 | 0 | — |
| Two genuine sentences exchanged within one article | 12 | 1 | — |

The comparison retained the wording of uniquely matched moved sentences.
It marked changes on one all-human sentence swap. These results measure
wording comparison with both versions available.

The same reader accepted all 25 previously saved publisher pages. Each
identified additional text outside its supported blocks. Comparing each
page with itself produced zero wording changes. A single-word replacement
inside each page mapped to the exact changed HTML ranges in all 25 cases.
The replay records hashes and ranges. Raw web snapshots and generated text
remain outside the repository.
[Case measurements](evaluation-panels/writing-html-draft-transfer-2026-11-18.json.gz)

The local app replay completed one simulated USD 9.99 analysis and queued
no job before payment. Submitting the earlier HTML created no extra analysis
job. The browser showed the skipped-text notice and four exact HTML ranges.
Its downloaded JSON and PDF matched the API exports. Offline verification
checked both original hashes and both changed words.
[Local replay receipt](evaluation-panels/writing-html-draft-app-2026-11-18.json)

## Reproduce

The numeric recount uses standard Python. The full transfer accepts the
retained input plan and publisher snapshots from the original writing
study. It binds those inputs to the earlier published numeric receipt.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.writing_html_draft_transfer review docs/evaluation-panels/writing-html-draft-transfer-2026-11-18.json.gz
```

Use the saved comparison JSON and the actual submitted versions for
original-file verification.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.review_document_revision_export chipmunk-verify-draft-comparison.json earlier.htm analyzed.html
```
