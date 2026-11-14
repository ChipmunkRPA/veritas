# Comparing separate sentence revisions

The optional earlier-draft writing check can now evaluate two to four
separate changed sentences in a short English TXT pair. Each sentence
must have one unchanged neighbor on each side in the same paragraph.
The neighborhoods cannot overlap. The check covers up to 420 words per
draft and requires every wording change to link to exactly one compared
sentence. It keeps the two uncalibrated before-to-after readings for
each sentence. Both must meet the unchanged paired rule of at least
double the AI-like odds for a shared increase. These results contribute
zero weight to authorship confidence.
[Prior paired rule and checkpoint terms](writing-paired-center-transfer-2026-11-12.md)

The browser, PDF and JSON show the same earlier and later positions and
link back to the exact wording changes. Each row states whether both
responses increased at the review level, the readings did not support
an increase, or a complete reading was unavailable. Positions are
zero-based Unicode code points with exclusive ends. The export verifier
rebuilds the locations and checks the decision arithmetic from the
saved responses.

## Frozen local comparison

We joined two three-sentence excerpts with a blank line. Each generated
rewrite kept its original neighbors from one of two existing GAO panels.
The other paragraph held an independently dated historical journal
revision with its own neighbors. An additional ten pairs joined two
ordinary journal revisions. This deliberately tests separate changes
while preserving the exact within-paragraph contexts. The composed
documents and decision rule were fixed before the paired models ran.
The evaluated prose had already been used in other project experiments.
Neither the composed passages nor the receipt substitute for a new
end-to-end trial on naturally revised multi-edit documents.
[First source panel](writing-context-margin-transfer-2026-11-13.md)
· [Second source panel and terms](writing-typed-decision-transfer-2026-11-13.md)
· [Historical journal versions](writing-journal-body-transfer-2026-11-11.md)

| Known change | Cases | Assessed | Exact increases |
| --- | ---: | ---: | ---: |
| Generated, first panel | 40 | 15 | 1 |
| Generated, second panel | 40 | 21 | 4 |
| Ordinary, distinct journal works | 21 | 11 | 0 |

The earlier optional check could assess none of the two-edit pairs. The
trial found five exact generated revisions across three source works and
both generators. There were 31 assessed generated revisions with no
shared increase and 44 unassessed revisions. Unsupported sentence
alignment or wording changes extending beyond a target make the pair
ineligible. The 0/11 ordinary result is specific to the assessable
subset, and the limited coverage keeps this a local trial.
[Source hashes, positions and responses](evaluation-panels/writing-separate-revision-transfer-2026-11-14.json)

One panel case went through simulated USD 9.99 checkout and the normal
local draft endpoint. No analysis jobs started before payment. With
the trial off, the paired check was out of scope. With it on, the two
sentence readings preserved their distinct outcomes. The analyzed
positions were [175, 336) and [747, 1049). PDF links and JSON replay
matched the source. Both exports omit checkpoint names.
[Local app receipt](evaluation-panels/writing-separate-revision-app-2026-11-14.json)

## Try and replay

Enable `VERITAS_DOCUMENT_PAIRED_SEPARATE_REVIEW=true` alongside the
earlier-draft comparison and optional writing-check settings. On a
retained, completed TXT analysis, sign in, open **Compare with an earlier
draft**, then select the optional writing check before submitting the
earlier TXT. Both drafts are required.

Replay a downloaded JSON against its two originals. The verifier
checks the ranges, links and recorded arithmetic without loading the
writing checkpoints.

```sh
cd backend
.venv/bin/python -B -m scripts.review_document_revision_export comparison.json earlier.txt later.txt
```

The research runner takes the retained local `roster.json` files from
the two linked source studies. The public receipt records their hashes
and contains no excerpt text.

```sh
cd backend
.venv/bin/python -B -m scripts.writing_separate_revision_transfer freeze "$source_rosters" "$freeze"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m scripts.writing_separate_revision_transfer \
  score "$source_rosters" "$freeze" --output "$receipt"
```
