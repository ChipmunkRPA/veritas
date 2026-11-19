# Numbered reference links in saved HTML

Chipmunk Verify can show when a bracketed citation number and the number printed
at its saved destination differ. The check reads standalone `.html` and `.htm`
files locally. Both positions are zero-based Unicode character ranges in the
decoded markup, with the end excluded. The signed package ties the ranges to the
submitted file. Authoring and conversion errors can explain a mismatch. This
observation carries zero weight in the AI assessment.

The check requires a bracketed decimal link in a saved paragraph and a uniquely
identified fragment within one explicitly numbered References list. The list
must contain at least eight entries numbered consecutively from 1. The reader
skips unresolved or ambiguous links, hidden markup covered by the static reader,
and documents that provide a base URL. The reader uses only the saved markup,
with zero external requests.

## Local evaluation

Six published PLOS ONE articles supplied saved HTML and corresponding JATS XML.
These articles were separate from the prior DOI comparison. The publisher uses numbered
in-text references and publishes its articles under CC BY 4.0. Publisher JATS
links supplied the original citation-to-entry correspondence. The manifest
was frozen before inspecting the alterations. [PLOS submission guidelines](https://journals.plos.org/plosone/s/submission-guidelines) · [PLOS terms and article license](https://journals.plos.org/plosone/s/licenses-and-copyright)

| Frozen HTML cases | Files | Exact discrepancies |
| --- | ---: | ---: |
| Changed the displayed citation number | 30 | 30 |
| Changed the saved link destination | 18 | 18 |
| Untouched publisher original | 6 | 0 |
| Ordinary prose addition with citations intact | 6 | 0 |
| Bracketed link to a numbered non-reference note | 6 | 0 |
| Duplicate reference identifier | 6 | 0 |

Across 370 eligible original reference links, the reader reproduced both
positions for all 48 authored changes and returned no discrepancy in the 24
controls. Each file contained one controlled change. The results cover six
articles from one publisher and measure a saved-link comparison. They leave
broader publisher layouts, linked claims, and text origin unmeasured.

The frozen source groups were PLOS ONE article identifiers `0335010`, `0345031`,
`0339027`, `0345601`, `0324906`, and `0358563`. On each, the selection sorted the
eligible native `(anchor, marker start)` pairs by SHA-256 of
`numbered-reference-link-held/1|<article identifier>|<anchor>|<marker start>`.
The first five supplied same-width citation substitutions. The first three
also supplied same-width destination substitutions. The frozen manifest hash
was `e7c5c6ed5d1f7620411a3c65d0c0092bfefa4f1c69fab9bc222e88620e58ec69`.

Five additional fixtures completed the local paid analysis flow with simulated
USD 9.99 checkout and zero jobs before payment. Both altered references kept
the same two offsets in Chromium, the API, PDF and JSON. Matching references,
hidden duplicate identifiers and numbered non-reference links produced no
finding. All five signed packages verified against the originals. Changing
only a saved destination left the overall writing assessment identical.
The redacted PDF withheld the finding.

A broader printed-text prototype found 21 of 22 controlled changes in both TXT
and DOCX. It flagged 6 of 6 ordinary numeric intervals and became ambiguous on
3 of 6 truncated bibliographies, so it stayed in evaluation.
Only two of six native PDFs had a complete numbered list that the prototype
could establish. Future coverage would need reliable saved link destinations
and source positions in each format.
