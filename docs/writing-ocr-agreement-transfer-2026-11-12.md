# Testing a second writing check on scanned PDFs

The proposed second writing check corroborated five known generated passages in 24 clear scans. The current scan check already located all five. The proposed check also corroborated a genuine edit after recognition changed a single en dash to a hyphen. The app continues to use the existing writing check on scanned text and preserves the page and word positions behind its responses. [Held receipts and source commitments](evaluation-panels/writing-ocr-agreement-transfer-2026-11-12.json) · [Existing scan evidence and source verification](writing-ocr-scan-transfer-2026-11-12.md)

## Frozen comparison

The source set uses 14 previously unscanned work groups from the earlier native-format panel. Each group supplies unchanged English prose and a same-work ordinary substitution. Four previously qualified generators supplied 24 known edits. The native prose and generated files had prior project exposure. The work groups and delivered scan hashes are disjoint from the 14-group scan extraction study. A separate check reopened all 52 original excerpts and reproduced the pixels of all 82 grayscale pages. The test uses attributable government prose and the same pinned commercial-use checkpoints as the native format study. [Source and model terms](writing-checkpoint-agreement-2026-11-11.md#frozen-inputs-and-comparisons) · [Native format commitments](writing-agreement-document-format-transfer-2026-11-12.md)

The pilot accepts one or two upright image-only PDF pages with complete English text recognition and per-page mean recognition quality of at least 95. Each excerpt has 68–420 extracted words in one or four paragraphs. A one-paragraph excerpt requires exactly four extracted sentences. Each second-check context stays on one source page. Its non-whitespace characters must map to the underlying word boxes, with recognition quality of at least 80 for each intersecting word. The second check reads up to the first six qualifying positive primary contexts in source order. Both checkpoints use their existing private decision points. The proposal carries zero weight in the whole-document assessment.

The useful-location rule requires at least 30% intersection over union and 50% recall of the known edit's non-whitespace characters. The comparison aligns the original and recognized characters, retains the whole known-edit denominator and treats unmapped truth as uncovered. The advancement criterion set before inference requires two useful work groups across two generator families, at most one additional genuine work group relative to native second-check results and complete source positions for all scored second-check contexts.

## Measured results

| Outcome | Existing check on eligible scans | Proposed second check on scans | Second check on selectable-text equivalents |
| --- | ---: | ---: | ---: |
| Known generated edits usefully located, out of 24 | 5 | 5 | 7 |
| Unchanged or ordinarily edited files flagged, out of 28 | 10 | 7 | 7 |
| Ordinary work groups flagged, out of 14 | — | 6 | 5 |
| Generated files flagged by the second check only at an insufficient location, out of 24 | — | 3 | — |

The source gate admitted 46 of 52 scans. Only 16 preserved the complete native transcript, despite the clear raster. The real classifiers read 297 primary contexts and 37 second-check contexts. Twenty files received at least one second response. Three work groups and three generator families supplied the five useful corroborations. The numeric gate passed. A normal PDF, rotated scan, mixed native and scan file, file with an unreadable page and low-quality scan stayed outside the proposed scope.

The follow-up isolated every corroborated scan context with one change from U+2013 to U+002D and an otherwise identical selectable-text context. Four CPU forwards per checkpoint measured the seven pairs. Three primary responses crossed the same decision point, including one genuine airport notice and two known generated edits marked at insufficient locations. For the genuine case the primary raw response moved from approximately 0.110 to 0.598. The second raw response was already positive on the exact original wording. This paired diagnostic tests the recognized punctuation and its effect on the two fixed checks. It supplies no tuned replacement or population error estimate.

## Replay

The portable reader reproduces the 52 outcomes, frozen context order, 334 exact response bindings and seven paired punctuation decisions with Python's standard library. The receipt contains hashes, anonymous work-group identifiers, source-page coordinates, overlap counts and saved scalar readings. Exact prose, model inputs and PDF fixtures stay local.

```sh
python3 -B -I -S backend/scripts/review_writing_ocr_agreement_transfer.py \
  docs/evaluation-panels/writing-ocr-agreement-transfer-2026-11-12.json
```

To rebuild the private input set, retain the `freeze.json` and `inputs` directories from the native-format run and the earlier scan freeze from the linked studies. Remove the earlier scan work groups from the native roster. Render each remaining selectable-text PDF with the `clear` profile in the [scan transfer runner](../backend/scripts/writing_ocr_scan_transfer.py). Verify the 52 exact PDF and text commitments, then run local extraction and segmentation. Use the app's [primary classifier](../backend/app/detectors/text/classifier.py) and [second-check runner](../backend/app/detectors/text/adapters/agreement.py) with the receipt's pinned checkpoint files and full-token, source-ordered microbatches of four. The native source audit separately checks the unchanged PDF geometry and exact pixels. The public command verifies the recorded outcomes from the available scalars.
