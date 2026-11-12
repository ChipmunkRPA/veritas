# Writing agreement across native document formats

All 100 held English excerpts preserved their exact text, ordered writing inputs and native source positions when delivered as DOCX, selectable PDF or plain-body email. The replay recovered the same 14 useful generated-edit locations and the same 14 human flags in each format. It used saved responses from the two pinned classifiers and made zero new inference calls. [Audited case commitments](evaluation-panels/writing-agreement-document-format-transfer-2026-11-12.json)

| Format | Human excerpts with exact source parity | Generated edits with exact source parity | Ordered contexts | Source span anchors | Omitted files |
| --- | ---: | ---: | ---: | ---: | ---: |
| TXT baseline | 56/56 | 44/44 | 683 | 1,849 | 0 |
| DOCX body paragraphs | 56/56 | 44/44 | 683 | 1,849 | 0 |
| Selectable PDF | 56/56 | 44/44 | 683 | 1,849 | 0 |
| Plain-body EML | 56/56 | 44/44 | 683 | 1,849 | 0 |

## Frozen sources and format constraints

The two retained writing panels supply 57 four-paragraph excerpts and 43 four-sentence excerpts. Their 28 source groups include 28 unchanged excerpts, 28 human edits and 44 generated replacements. All source text hashes and prior response commitments come from the pinned agreement comparison. Repackaging leaves the held language and exact known edit boundaries intact. The test uses new file containers and reuses already evaluated prose. [Source attribution, checkpoint terms and original outcomes](writing-checkpoint-agreement-2026-11-11.md)

Each DOCX contains top-level body paragraphs with multiple style runs. Each native PDF uses selectable text and an embedded font. Four-paragraph PDFs put two whole paragraphs on each page. Four-sentence excerpts stay in one paragraph. Each EML uses a single UTF-8 plain-text body with quoted-printable transport encoding. The fixtures preserve curly apostrophes, en and em dashes, soft hyphens, section signs and copyright symbols present in the retained corpus. All tested document files were constructed without customer text.

For DOCX, an independent XML read verified 2,772 run and decoded-range positions. For PDF, native word extraction verified 16,651 boxes against the character-level source mapping. For EML, independent MIME decoding verified 271 exact decoded-body ranges. The analysis's normal source-anchor helper resolved each scored span to its native paragraph, word boxes or body range. The exact neighboring-sentence input hashes, offsets, stored window selection and secondary-check direction agreed with TXT. The original file SHA-256 and extracted text SHA-256 were checked separately.

## Excluded controls

The proposed source gate rejected all eight quoted plain emails and eight HTML-only emails. The HTML controls decode to exactly the same normalized words. They remain outside this tested source scope because their text maps to markup. Eight image-only PDFs required OCR and could not enter the native text comparison. Tables, headers, signatures, quoted email passages, HTML bodies and OCR need separate evidence for any future expansion. The agreement check continues to carry zero authorship confidence weight.

The result supports format equivalence for these controlled layouts. It supplies no new evidence about detection accuracy or arbitrary document reading order.

## Local worker and exports

One held generated-edit excerpt ran through the signed test checkout and the real CPU worker in all four formats. Checkout and unpaid events queued no analysis. Each paid test event allowed its analysis to run. The worker reproduced the same second response, three input spans, and corroborated focus at character range **[372, 516)**. The evidence API, JSON and PDF matched those results. Independent replay matched the two PDF source crops to the original page pixels and verified the report's embedded images. The Word and email exports identified the original body paragraphs and MIME body part. Quoted and HTML-only email and a scanned PDF completed their ordinary analysis with the second check omitted.

## Reproduce

Use the existing pinned agreement and parent panel directories as `$panels`. The fixture renderer uses the locally installed DejaVu Sans file whose SHA-256 is in the receipt. Set `$run` to a new private directory. The generated files and source prose remain local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.writing_agreement_document_format_transfer freeze "$panels" "$font" "$run"
backend/.venv/bin/python -B -m scripts.writing_agreement_document_format_transfer run "$panels" "$font" "$run"
python3 -B -I -S backend/scripts/review_writing_agreement_document_format_transfer.py "$run"
```

The no-media replay checks all 400 comparison commitments and 24 controls with the Python standard library.

```sh
python3 -B -I -S backend/scripts/review_writing_agreement_document_format_transfer.py \
  docs/evaluation-panels/writing-agreement-document-format-transfer-2026-11-12.json
```
