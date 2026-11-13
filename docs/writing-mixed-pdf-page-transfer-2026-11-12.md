# Second writing check on selectable text in mixed PDFs

A Full or Deep local development run can now revisit selectable-text passages in a PDF with an unchecked or recognized page elsewhere. Each retained context stays entirely on one intact selectable-text page. The ordinary evidence view links the passage to that page’s original crop and highlight. The Standard PDF gives the same page number, focus and context positions, and the JSON retains the exact source ranges and input commitments.

The feature uses the existing optional second writing check. Its scope is one complete four-sentence English text block or four complete English text blocks, with 68–420 words in the eligible blocks combined. It revisits the first six eligible AI-like primary contexts in source order. Every retained native word has to map to a bounded source box. All pages with an OCR record or a partially checked image area are outside this check, including their selectable captions. Contexts that cross pages are also outside it. The second check carries zero weight in the overall assessment. [Existing page-by-page coverage](writing-pdf-page-coverage-2026-11-12.md) · [Earlier scanned-text agreement test](writing-ocr-agreement-transfer-2026-11-12.md)

## Controlled local comparison

The paired layout study inserted one page after the first source page in each of 100 pinned selectable-text PDFs. The inserted page was blank in 50 files and contained an image with no text in 50. The 28 work groups included 44 known replacements from four generator families and 56 unchanged or ordinarily edited controls. The prose and edits had prior project exposure. The PDF layouts are new, and the study preserves the same exact native text in each pair. The licensed checkpoints and source terms come from the earlier agreement and format studies. Research on hybrid writing motivated retaining exact local context and source positions. The implementation uses the app’s own extraction, context rules and licensed weights. [Source and checkpoint terms](writing-checkpoint-agreement-2026-11-11.md#frozen-inputs-and-comparisons) · [Original PDF commitments](writing-agreement-document-format-transfer-2026-11-12.md) · [MixDetect](https://arxiv.org/html/2609.32625v1) · [LA-CPD](https://arxiv.org/html/2609.33787v1)

The runner verified all 683 ordered first-check input hashes and 16,651 selectable-text word positions before reusing the corresponding first-check responses. A separate CPU pass measured the second check on 42 distinct ordered requests across the two layouts, for 48 model forwards and 102 measured context inputs. The layout scope admitted 69 context occurrences and kept 114 cross-page inputs outside the second check. A useful location needed at least 30% intersection over union and 50% recall of the known edit’s nonwhitespace characters. The criterion required four useful locations across at least two source groups, no new genuine group flagged relative to the intact layouts, and complete native positions for every retained context.

| Outcome | Previous second check on mixed PDFs | Same-page second check on mixed PDFs | Second check on intact equivalents |
| --- | ---: | ---: | ---: |
| Known edits usefully located, out of 44 | 0 | 11 | 14 |
| Genuine or ordinarily edited files receiving agreement, out of 56 | 0 | 13 | 14 |

The previous route skipped the entire mixed file, so its zero counts describe abstention. The new scope found useful evidence in eight work groups and added no genuine group flagged relative to the intact equivalents. The 13 genuine responses show why the agreement is a lead for review and carries zero confidence weight. This comparison measures restored local coverage in controlled layouts. It does not estimate accuracy on customer files or establish accuracy for OCR text.

## App verification

Five isolated local jobs went through the simulated payment flow. Unpaid events queued zero jobs. The installed classifiers produced corroborated passages on page 3 of one edited PDF and page 1 of another. One genuine control retained agreement and the other received no agreement. A scan-only control received no second check. The API and JSON agreed on all passage ranges, and the Standard PDF carried their page numbers and response labels. The eight source crops and highlight pairs matched the committed original page positions. Chrome followed a corroborated passage link and loaded the page 3 images in the ordinary evidence view. Separate tests with successful local OCR and with an unchecked image area both kept the first page outside the second check and retained the intact second page. [Local app receipt](evaluation-panels/writing-mixed-pdf-page-transfer-app-2026-11-12.json)

The signed mixed-PDF package replayed against the original and rejected a changed original and a changed page-scope record. An earlier signed r107 package also replayed against its own original.

## Replay

The public reader reproduces the comparison from hashed inputs, ordered requests, exact response values and overlap counts. The receipt excludes source prose, PDF fixtures and model weights. [Panel and measured outcomes](evaluation-panels/writing-mixed-pdf-page-transfer-2026-11-12.json)

```sh
python3 -B -I -S backend/scripts/writing_mixed_pdf_page_transfer.py review \
  docs/evaluation-panels/writing-mixed-pdf-page-transfer-2026-11-12.json
```

With the retained earlier panel directories and pinned checkpoint available, run the same script with `freeze` then `measure`, using the same new scratch directory and `--panels` root for both. Supply the local pinned model snapshot with `--checkpoint` for `measure`. Set `PYTHONPATH=backend`, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for both stages.
