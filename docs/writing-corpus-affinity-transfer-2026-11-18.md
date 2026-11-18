# Testing phrase overlap for localizing AI rewrites

An independent local test asked whether a compact index of ordinary phrases
could help locate a short AI rewrite inside a mostly human document. The index
used up to six consecutive words from 800 pre-2021 GovReport excerpts. The
index, calibration and evaluation source documents were disjoint. The test
looked at low overlap with the index and at sentences whose overlap was low
relative to the rest of the same document. This is a small adaptation of
research on how closely generated writing resembles ordinary corpora.
[EMNLP research](https://aclanthology.org/2026.emnlp-main.708/)
· [GovReport terms](https://gov-report-data.github.io/#license)

The held set comprised 129 files across three collections. It contained 60
labeled rewrite locations in 42 edited files, 46 untouched files and 41 files
with ordinary substitutions. The experiment separately calibrated both
variants on 109 ordinary source groups. A finding needed to locate the
labeled writing span. The ordinary-file counts below concern passage
highlights.

| Outcome | Existing passage checks | With raw phrase overlap | With within-document comparison |
| --- | ---: | ---: | ---: |
| Known edits located | 2/60 | 3/60 | 2/60 |
| Untouched files with a highlighted passage | 12/46 | 14/46 | 12/46 |
| Ordinary substitutions with a highlighted passage | 9/41 | 11/41 | 9/41 |
| Additional misplaced highlights in AI-edited files | — | 11 | 2 |

Both variants ranked the correct sentence first for only 2 of 60 edits.
Fifty-seven edits overlapped a sentence that the experiment could score.
The evaluation supports keeping both variants experimental. A second run
reproduced the numeric files byte for byte. The published summary contains
counts, controls, and input commitments.
[Numeric summary](evaluation-panels/writing-corpus-affinity-transfer-2026-11-18.json)
