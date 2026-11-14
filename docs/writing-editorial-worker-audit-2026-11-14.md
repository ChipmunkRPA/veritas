# Single-upload writing findings on ordinary journal revisions

We ran each excerpt as a separate Full document analysis through the local
Apple GPU worker. This exercised the same saved results, customer evidence
cards and JSON export as a normal upload. Each case used a simulated US$9.99
checkout. The run checked payment before execution, idempotent fulfillment,
the exact hashed source and input ranges, and parity between the evidence API
and the JSON export. The worker used the existing serving checks and settings.
The app and Standard PDF now explain beside the passage evidence that ordinary
editing and polished writing can produce similar responses.

## Inputs and measurement

The inputs are the exact 74 texts from the previous Wellcome Open Research
and Gates Open Research journal body panel. The original benchmark selected
dated 2018–2019 CC BY 4.0 article versions and retained one changed middle
sentence flanked by two unchanged sentences on each side. The 21 author work
groups contributed their dated earlier and naturally revised excerpts. The
same source groups supplied 16 usable Qwen and 16 usable SmolLM2 rewrites of
the middle sentence. Both generator snapshots carry an Apache-2.0 license.
The two journals share a publishing platform. The source excerpts were
previously evaluated by paired and proxy methods. Prior checkpoint training
exposure is unknown. [Source freeze and checkpoint terms](writing-journal-body-transfer-2026-11-11.md)

An evidence selection means at least one passage card appeared. A useful
sentence location means one marked span covered at least half the known middle
sentence with a non-whitespace character intersection over union of at least
30%. The earlier version's middle sentence is a positional comparison. The
analysis also tracks the card's navigation sentence so it can identify it
appearing on the same unchanged sentence in both versions. A separate lexical
recount uses case-folded word alignment to map inserts and replacements in the
middle sentence against its earlier version. A word counts as covered when
every character falls in the union of the marked spans. Removed words are
outside the submitted excerpt. The word alignment measures coverage of changed
wording. Authorship of individual words remains unknown. The panel measures
short, preselected English article excerpts and cannot estimate customer-wide
performance. [Peer-review polishing study](https://arxiv.org/abs/2603.20450)

## Results

Both published writing checks completed on all 74 inferences. Every
whole-document decision was inconclusive. Passage cards appeared frequently
on both ordinary and rewritten excerpts.

| Submitted excerpt | At least one card | Known middle sentence covered | Cards only elsewhere | Inconclusive |
| --- | ---: | ---: | ---: | ---: |
| Earlier publisher version | 17/21 | 10/21 | 7/21 | 21/21 |
| Ordinary revised publisher version | 16/21 | 6/21 | 10/21 | 21/21 |
| Qwen rewrite | 15/16 | 9/16 | 6/16 | 16/16 |
| SmolLM2 rewrite | 15/16 | 9/16 | 6/16 | 16/16 |

At least one of the two ordinary uploads selected evidence in 17 of 21
article pairs. In 13 pairs both uploads selected the same unchanged navigation
sentence. Three later uploads added a new navigation sentence, and only one
of those added the changed sentence. The interface therefore treats a marked
passage as a review lead. A rendered Standard PDF from a naturally revised
excerpt showed an inconclusive assessment and the new disclosure next to its
one passage selection on the document-evidence page.

| Submitted revision | At least one changed word covered | All changed words covered, pooled by word count | Marked characters in changed words |
| --- | ---: | ---: | ---: |
| Ordinary publisher revision | 4/21 | 12/63 | 67/3,091 |
| Qwen rewrite | 9/16 | 82/145 | 493/3,775 |
| SmolLM2 rewrite | 9/16 | 101/188 | 568/3,524 |

The second column counts files. The third counts individual fully covered
words across each cohort. The fourth measures how much of the displayed
highlight falls on changed words, including partially covered words. Five
ordinary later versions had no inserted or replaced words under this alignment
because deletions, punctuation, or case changes can qualify as a material
sentence difference. Even on the known rewrites, only 1,061 of 7,299 marked
characters fell inside inserted or replaced words. This run leaves the
serving checks and combined confidence unchanged.

## Replay

From the repository root, use standard-library Python to recount the public
evidence positions, source bindings and word overlap. The receipts preserve
offsets, text hashes and observed outcomes. Excerpts and model weights remain
local. Native replay requires the exact retained input file and the installed
worker models on Apple Silicon. No accounts, live payments or hosted inference
are required for the numeric replay.

```sh
export PYTHONPATH=backend
worker=docs/evaluation-panels/writing-editorial-worker-audit-2026-11-14.json
words=docs/evaluation-panels/writing-editorial-word-overlap-2026-11-14.json
python3 -B -S -m scripts.writing_editorial_worker_audit review "$worker"
python3 -B -S -m scripts.review_writing_editorial_word_overlap review "$words" --worker "$worker"
```

Point `journal_cases` to the original `scored/cases.json` from the pinned study
and supply an empty path for `run_dir` to regenerate the receipts.

```sh
python3 -B -S -m scripts.writing_editorial_worker_audit preflight "$journal_cases"
backend/.venv/bin/python -B -m scripts.writing_editorial_worker_audit run "$run_dir" --source "$journal_cases"
python3 -B -S -m scripts.review_writing_editorial_word_overlap run "$new_words" \
  --worker "$run_dir/receipt.json" --source "$journal_cases"
```

After an interrupted native run, keep its output directory and pass `--resume`
to the same worker command. The retained progress must match the ordered input
prefix. This panel retained its original results through a local storage
interruption and resumed on the same worker configuration.
