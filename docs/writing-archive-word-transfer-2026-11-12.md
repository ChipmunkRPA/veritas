# Exact archive changes in Word documents

The public archive comparison now accepts `.docx` in local development. It cites the matched archival excerpt and shows the exact wording changes, their Word body paragraph numbers and their character ranges. The JSON and API also include the exact text-run fragments for each changed token when the Word run mapping is complete. This source comparison carries zero weight in the AI assessment. [User workflow](writing-public-archive-comparison.md)

## Held check

The held panel included 96 native Word documents. They use retained publisher text and exact generated or ordinary edits whose source locations were already committed in the earlier plain-text study. The layout added a blank top-level paragraph, expanded horizontal spaces and split native text into 23-codepoint runs. This placed some changed words across runs and made the paragraph-relative ranges differ from normalized analysis offsets. The test measures source-position transfer to new containers using previously measured text. [Diplomatic source and edit qualification](writing-archive-source-evidence-2026-11-12.md) · [Kubernetes source qualification](writing-multiparagraph-localization-transfer-2026-11-12.md) · [Federal Register source qualification](writing-practical-notice-transfer-2026-11-11.md)

| Held Word input | Documents | Outcome |
| --- | ---: | --- |
| Indexed diplomatic originals | 16 | 16 correct source matches, zero differences |
| Indexed originals with an ordinary edit | 16 | 16 correct source matches with exact changed ranges |
| Indexed originals with a known generated replacement | 21 | 21 correct source matches with exact changed ranges |
| Out-of-index diplomatic, Kubernetes and Federal Register writing | 37 | Zero source claims |
| Indexed source text in tables only | 6 | Zero source claims |

The replay verified 545 exact changed-token positions, including 97 tokens spanning more than one Word run. It reopened `word/document.xml` independently and reconstructed each raw paragraph and run fragment. All source citations, extracted coordinates and text differences agreed with the corresponding plain-text comparisons. The held test leaves real-world document accuracy and AI authorship outside its scope. [Public receipt](evaluation-panels/writing-archive-word-transfer-2026-11-12.json)

## Local verification

An isolated Full document analysis ran the normal worker with the lightweight writing lane. The simulated unpaid event queued zero jobs, the paid event queued one, and a repeat of the paid event added zero jobs. Chrome’s ordinary result showed the cited source, the Word paragraph number and range, and the original and changed word. The downloaded JSON matched the API. The downloaded PDF matched the API bytes, and a separate PDF read confirmed the same wording and native range. The route returned a readable error for malformed Word input. The Word archive comparison uses the retained original and the top-level body scope.

Use the retained diplomatic source directory, the retained Kubernetes multi-paragraph run and the Federal Register run on the development archive. Replay uses no model weights. The frozen result records the renderer and source commitments.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
backend/.venv/bin/python -B -m scripts.writing_archive_word_transfer \
  freeze "$run/freeze.json" "$archive" "$articles" "$notices"
backend/.venv/bin/python -B -m scripts.writing_archive_word_transfer \
  verify "$run/freeze.json" "$archive" "$articles" "$notices"
```
