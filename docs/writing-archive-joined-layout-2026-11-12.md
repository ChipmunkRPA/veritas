# Archive evidence when a text copy loses paragraph breaks

The existing source comparison depends on paragraph boundaries in the submitted text. A new candidate recovered every source on 16 held unmodified excerpts joined into one paragraph and the exact edited word in all 16 matched replacements. The existing comparison skipped all 32 one-paragraph views. The candidate made no incorrect source claims among 71 controls and reported zero differences on all 16 unchanged excerpts.

| Held input | Files | Correct source matches | Useful edit locations |
| --- | ---: | ---: | ---: |
| Unchanged excerpts with original paragraph breaks | 16 | 16 | — |
| Same excerpts with paragraph breaks collapsed | 16 | 16 | — |
| Collapsed excerpts with a known same-work word substitution | 16 | 16 | 16 exact |
| Reordered, mixed-source, prefixed, quoted, ambiguous or out-of-index copies | 71 | — | 0 claims |
| Earlier generated sentence replacements, newly collapsed | 21 | 10 | 10 overlapping |

## Source comparison

The study used the existing library of selected U.S. State Department archival excerpts. The held group selection excluded prior queried works and exact 12-word overlaps. The 16 held works have distinct official sender keys within this panel. Some of those institutions or individuals occur in earlier studies. The 21 generated sentence replacements are separately reported because their source works and rewrite text were tested earlier. Source, changed-word, and edit-boundary positions were fixed before scoring. [Archive intake and license](writing-archive-source-evidence-2026-11-12.md)

The candidate matches the exact first and last eight tokens of three or more consecutive reference paragraphs in a single extracted plain-text paragraph. It requires a unique chain covering all submitted tokens. The existing fingerprint check then confirms the source and each recovered boundary. The existing exact word and punctuation comparison measures changes within those spans. The coordinates stay in the submitted file's normalized extracted text. This approach can abstain when an edit touches a boundary, when surrounding material is added, or when the relevant source is outside the index. It contributes zero AI-authorship weight. [Source-alignment task](https://pan.webis.de/clef25/pan25-web/generated-plagiarism-detection.html)

The method replay reproduced all 140 frozen decisions. A separate audit verified 348 exact source spans and 119 changed-token positions, and rejected a one-code-point alteration. [Public decision receipt](evaluation-panels/writing-archive-joined-layout-2026-11-12.json)

## Replay

Use the Git commit that introduced this evaluation and the local parent folder from the [archive-source study](writing-archive-source-evidence-2026-11-12.md). Use a new output folder for `run` and retain the pinned XML volumes and prior study fixtures. The `replay` phase checks the measured output against the reconstructed inputs. The separate auditor independently checks exact source and changed-word positions.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
for phase in freeze measure replay; do
  backend/.venv/bin/python -B -m scripts.writing_archive_joined_layout_transfer "$phase" "$parent" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_writing_archive_joined_layout "$parent" "$run" "$run/audit.json"
```
