# Second writing check across additional email reply layouts

Reformatting 100 held excerpts into two- and three-paragraph email replies preserved 542 of 583 exact classifier inputs in each layout. Four variants from one source group changed sentence boundaries. Six-paragraph replies preserved all 440 inputs from 53 eligible excerpts. The broader layout proposal missed its frozen coverage gate, so this test makes no serving change. [Aggregate and source audit](evaluation-panels/writing-reply-paragraph-transfer-2026-11-12.json)

## Frozen comparison

The current optional second writing check accepts one mapped four-sentence paragraph or four mapped body paragraphs in supported plain-text email replies. The proposal keeps the same language, length, context and trailing quotation rules and tests two, three, five and six paragraphs. Each email contains a text reply, signature, quotation, alternate HTML body and unrelated fixture attachment. Distribute existing exact sentences as evenly as possible, preserving their order. Omit a planned format when an excerpt has fewer sentences than the requested paragraph count.

The source roster has 56 ordinary and 44 generated-edit excerpts across 28 previously evaluated groups. Replay the pinned primary and second responses only when the newly extracted body supplies the same ordered sentence text and exact context hashes. Check model inputs that stay wholly inside the reply. Compare their source localization with the same body-only input selection before reformatting. No new inference ran. [Source and classifier-response commitments](writing-checkpoint-agreement-2026-11-11.md) · [Current mapped reply support](writing-email-body-scope-transfer-2026-11-12.md)

The gate requires 96% exact input parity per layout, retention of 90% of useful comparison highlights, no additional human highlights and exclusion of all 24 HTML-only and interleaved quotation controls. A source setup attempt stopped when its exact-sentence assertion encountered a changed boundary. The completed freeze records any such delivery as unavailable for response replay and includes every planned input in the corresponding coverage denominator.

## Held results

| Body paragraphs | Repackaged files | Exact input parity | Useful generated locations, same-source comparison → reformat | Human files highlighted, same-source comparison → reformat |
| --- | ---: | ---: | ---: | ---: |
| 2 | 100 | 542/583 | 14 → 13 | 13 → 11 |
| 3 | 100 | 542/583 | 14 → 13 | 13 → 11 |
| 5 | 55 | 407/448 | 12 → 11 | 10 → 8 |
| 6 | 53 | 440/440 | 12 → 12 | 10 → 10 |

All 308 delivered replies had complete mapped plain-text body roles. The four variants with changed sentences at two, three and five paragraphs came from the same source group. This experiment withholds every saved response for those variants, so the lower human highlight counts include reduced response coverage. The 24 controls stayed outside the reply check. A future comparison can run the actual classifiers on those reformatted inputs and independently verify how changed segmentation affects both findings and source coordinates.

## Replay

Use the retained writing evaluation source directory as `panels` and an empty destination path as `run`. Run the exact study and the independent MIME reader. The public result contains aggregate counts and commitments without source prose.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.writing_reply_paragraph_transfer freeze "$run" "$panels"
backend/.venv/bin/python -B -m scripts.writing_reply_paragraph_transfer run "$run" "$panels"
backend/.venv/bin/python -B -m scripts.review_writing_reply_paragraph_transfer "$run" "$panels" "$receipt"
```
