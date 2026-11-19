# Can omitting one sentence locate a rewrite in a single upload?

The test applied a fixed local writing check to each three-sentence
window and to the same window with each sentence omitted in turn. A
sentence could rank first only if removing it consistently lowered the
reading more than removing its neighbors. The proposed evidence location
was that sentence alone. The prior isolated/context interval check and an
ordinary classifier-card reconstruction provided two comparisons. These
readings can share errors and carry no additional confidence weight.
[Feature-removal attribution method](https://jmlr.org/papers/v22/20-1316.html) ·
[Passage localization research](https://aclanthology.org/2024.findings-acl.495/)

The frozen source split used pre-2023 Federal Register text from eight
agencies for development and 17 works from 14 separate agency labels for
held testing. Each held work had an untouched copy and a conventional
same-article substitution. Nine one-sentence rewrites from two local
generators passed the prespecified shape and digit checks. Three
preserved the source meaning and six changed a factual qualifier. All
other sentences stayed identical. [Federal Register collection](https://www.govinfo.gov/help/fr)

| Held reading | Exact generated locations | Ordinary groups with an alert |
| --- | ---: | ---: |
| Existing optional interval check | 0 / 9 | 1 / 17 |
| Sentence omission | 0 / 9 | 3 / 17 |
| Either reading | 0 / 9 | 4 / 17 |

The sentence-omission check flagged two rewritten files, both at unchanged
sentences already flagged in their untouched counterparts. Its top sentence
remained at the same index as in the untouched document for seven of nine
rewrites. Even without the operating point, its top-ranked sentence matched
one of the nine interventions. The local outputs and independent recount
agree. The method stays offline.

The result covers fixed-length English excerpts and two generators. It
does not establish general authorship accuracy. Future evaluations should
include unchanged neighbors as explicit negative examples and test any
new selection on untouched sources. [Anonymous aggregate](evaluation-panels/writing-sentence-omission-2026-11-19.json)
