# Local scanned-PDF and email evidence

The document extractor preserves the existing PDF text-layer path. A page with
no usable text layer can be transcribed locally; text-layer pages are never
replaced with OCR guesses. No document is sent to an external OCR service.

The local runtime uses Tesseract 5 and the provisioned English, simplified Chinese,
traditional Chinese, and orientation data. `models/ocr-manifest.json` records the
official immutable source tree, upstream Git blob hashes, download URLs, sizes,
and SHA-256s. Inference verifies the local hashes and never downloads anything.
The language data live under ignored `models/weights/ocr/tessdata/`. `langid==1.1.6`
is installed and declared as a backend dependency for heuristic language coverage.
Another machine needs the executable and the pinned data provisioned locally;
otherwise scanned pages explicitly remain unchecked.

For each attempted page, the frozen decode records:

- Original page number, dimensions and PDF rotation.
- Exact PNG bytes supplied to recognition, committed to content-addressed storage
  before text scoring and indexed among the decode stage's dependent artifacts.
- Measured raster dimensions and DPI, grayscale conversion, orientation correction,
  and affine mapping from input pixels to unrotated PDF points.
- Recognized text, per-word input boxes and page boxes, Unicode-codepoint offsets,
  recognition confidence, language estimate, and a page status/reason.
- The OCR engine and language-data hashes internally. Public evidence omits those
  implementation names while retaining the raster/text/geometry commitments.

Recognition confidence measures transcription quality. It is never interpreted
as AI-generation confidence. Unknown or unsupported language, insufficient text,
low recognition quality, a timeout, or an exhausted budget leaves the page
unchecked and excludes its guessed text from classifier input. English and
Chinese identification remains heuristic; this does not establish that the
authorship detectors have been validated for every language or document style.

Bounds are 12 OCR pages, 4 million raster pixels per page, 24 MiB frozen PNGs,
and 70 seconds of OCR work within the existing extraction deadline. Each OCR
subprocess has its own timeout; the outer extraction timeout kills its process
group, including OCR descendants. Budget exclusions are recorded as unchecked.
Documents with too little reliably extracted text continue to fail as unassessed.

For partially readable PDFs, the whole-document decision and probability are
withheld on both fallback and calibrated fusion paths. Measured signals and
sentence/context results remain available for the extracted text. This prevents
unchecked pages being interpreted as evidence of human authorship.

`chipmunk.document-evidence/1` remains additive. OCR anchors use
`extraction: "ocr"`, carry recognition metadata, and expose exact selected word
boxes. Existing context/crop/overlay artifacts derive from the frozen OCR raster,
not a fresh OCR pass. The additional `pageN_ocr_input` PNG is the exact recognition
input, byte for byte. Ordinary PDF text-layer evidence keeps its existing crops.
Resuming a decode requires the current extraction version and all frozen raster
dependencies; old decode artifacts are not silently reinterpreted.

EML is accepted by the existing quick-upload, case-intake and batch paths under
the document upload limit, retaining `.eml` and `message/rfc822`. Extraction
prefers one plain MIME body over its HTML alternative, retains quoted text and
signatures with heuristic labels, and does not authenticate headers or senders.
Attachments remain outside the selected body and are explicitly unexamined.

Focused validation includes real local English OCR, a Chinese smoke document,
scanned/rotated/mixed pages, all four affine orientation cases, Unicode offsets,
low-confidence/unknown-language/timeout exclusions, exact frozen raster crops,
decode resumption dependencies, partial-scope verdicts, actual classifier window
binding, and real quick-upload/case-intake EML storage and size limits.

## Opposing text-check contexts

Document pipeline `doc-2.0-p20` records the paired zero-shot check's actual token
inputs as `scored-context/2`. The established learned-classifier
`scored-context/1` sentence windows remain unchanged. New public packet sets use
`document-evidence-2`; historical frozen sets retain their bytes and version.

Each rolling inference context records the exact input token IDs (including an
optional leading anchor), their canonical SHA-256, Unicode-codepoint token
offsets, half-open source/scored token ranges, the 1024-token context limit and
768-token stride. Both paired tokenizers must have identical vocabulary IDs
and actual source tokenization/offsets. Their default end-token IDs may differ;
the first tokenizer's explicitly chosen leading anchor is supplied to both
models in the same input sequence. Both defaults and that actual anchor choice
are recorded. Text in the context is the
source envelope of those tokens, not a claim that re-tokenizing a displayed
substring recreates the original input. No model inference occurs on reads.

Each sentence response identifies its reduction span, selected token indices,
loss means, raw ratio and frozen threshold. Short sentences can use their
neighbours; `neighbor_expansion` exposes that existing scoring behaviour. Exact
hexadecimal values preserve threshold direction and loss means across the
canonical JSON serializer's six-decimal display rounding. A ratio can exceed 1;
lower is on the configured AI-like side. It is not a calibrated probability.

Selected packets retain both supporting and opposing check responses, each with
its own score kind, operating boundary and direction. Selection reserves up to
half the six displayed passages for disagreement, fills other slots from the
remaining candidates, and uses source order within each group. This prevents
corroborated passages crowding out actual opposing observations and does not
pool incomparable ratios and classifier outputs. Direction only describes
which side of an operating threshold the measured response falls on; it is not
validated authorship evidence.

Token-context recording is bounded to 262144 source tokens and 16 input contexts
per local response, with at most six selected packets. Exceeding an evidence
bound does not change inference: the affected local input proof is explicitly
omitted. Insufficient-token omissions and the unscored initial token when no
anchor exists are also recorded. `coverage.checks` distinguishes recorded,
partial and absent local context proof for each successful check. Stylometric
document aggregates and derived sentence consensus do not become independent
localized evidence; their missing context proof remains an explicit limitation.
The absence of stylistic flags is not fabricated as an opposing check result.

`document_packets.verify_context_window` checks exact source excerpts, token
input hashes/offsets, reduction membership, recorded means and threshold direction
without re-tokenizing or running inference. Tests cover rolling boundaries,
Unicode offsets with overlapping token pieces, no-anchor behaviour, neighbour
expansion, mismatched paired tokenizers, ratios above 1, opposing classifier
results, canonical rounding, bounded omissions, and tampered source/token records.
