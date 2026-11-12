# Replay writing evidence from scanned PDFs

The source verifier can reopen a scanned or mixed PDF from a signed evidence package. It re-runs local text recognition, checks the exact extracted text and word positions, and compares the recognition raster and evidence crops against the supplied original. The replay checks the evidence behind the report. Authorship scoring is outside its scope.

Run the verifier with the project's pinned OCR language data and a compatible PDF renderer and recognition runtime. A difference in the recognized text, mapping, or original raster makes the source check fail. The package signature can still be checked on its own. OCR names and installation identities remain absent from public analysis exports. [Pinned OCR files](../models/ocr-manifest.json) · [OCR data license](https://github.com/tesseract-ocr/tessdata_fast/blob/87416418657359cb625c412a48b6e1d6d41c29bd/LICENSE)

```sh
python -m app.cli verify evidence.zip --public-key instance-key.pem --source-document original.pdf
```

## Scan transfer

The panel contains 14 previously tested source groups. Each group supplies the same unchanged excerpt, a human splice and one known generated replacement as the earlier format study. Each of these 42 English excerpts was delivered as a clean 200 dpi grayscale scan and a 110 dpi lower-contrast JPEG scan. Every delivered page contains only a raster image. The scan delivery and source commitments were frozen before extraction. The source prose and the earlier text responses had prior project exposure. The scan experiment made zero authorship inference calls. [Earlier text and format qualification](writing-agreement-document-format-transfer-2026-11-12.md)

| Per scan condition | Clear scan | Lower-quality scan |
| --- | ---: | ---: |
| Fully extracted documents | 42/42 | 42/42 |
| Reproduced full source text and ordered inputs exactly | 11/42 | 5/42 |
| Identical classifier inputs anywhere in the same document | 168/289 | 121/289 |
| Word insertions, deletions or substitutions over 6,934 source words | 60 | 92 |

The native transcript and scan transcript can differ in punctuation, spacing, paragraph separation, or words. A small transcription change can affect several overlapping classifier inputs. Readable pages and a high aggregate OCR quality do not establish that the original classifier inputs survived. The second writing check for native documents stays outside this OCR evidence replay. The controlled files do not estimate performance on handwriting, multi-column pages, customer scans, or other languages. [Aggregate and per-group counts](evaluation-panels/writing-ocr-scan-transfer-2026-11-12.json)

## Local verification

Two scans completed the ordinary paid analysis flow using simulated payment and the real writing classifier. The PDF, JSON, evidence API and signed package agreed on three passages per scan. A separate page render matched nine crop placements across the two reports. The CLI accepted both signed packages with their originals and rejected changed originals. Signed rotated scanned and hybrid fixtures also exercised the exact OCR input, deliberately altered recognition metadata, and unavailable local OCR runtime. [Local verification](evaluation-panels/writing-ocr-scan-app-2026-11-12.json)

## Reproduce

Use the retained native format panel as `formats` and a new private directory as `run`. The script renders the images and measures the exact text and input hashes locally. The published receipt includes counts without the files or their text.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1
for phase in freeze measure; do
  backend/.venv/bin/python -B -m scripts.writing_ocr_scan_transfer "$phase" "$run" "$formats" || exit 1
done
python3 -I -S backend/scripts/writing_ocr_scan_transfer.py review "$run/result.json"
```
