# Invisible characters in uploaded text

Chipmunk Verify now records selected invisible Unicode characters in
decoded TXT, mapped Word body and table paragraphs, and the selected
plain-text or source-mapped HTML email body. The report shows the character
name and code point with its exact decoded source range. For example,
a mapped HTML entity keeps the full source range of the entity and one
position in the extracted text. The original bytes and extraction used
by writing checks stay intact.

The check records left-to-right and right-to-left overrides, embedded
byte order marks, and three invisible breaks or joiners when they
appear inside runs of basic Latin letters and digits. A leading
encoding signature, ordinary direction isolates and join controls used
in scripts or emoji are outside the matching rule. The display shows
up to 24 locations and counts the rest. Positions use Unicode code
points, start at zero, and have exclusive ends.
[Unicode bidirectional formatting specification](https://www.unicode.org/reports/tr9/)

These characters can come from ordinary layout, accessibility tools,
or copy and paste. The observation contributes zero authorship
confidence. It does not verify a provider watermark. PDF glyph text,
unmapped HTML, email attachments and unextracted Word content are
outside this check. Checkout permits the recognized formatting controls
when the file is otherwise readable and rejects a document made only
of invisible formatting.

## Verify the record

Run the optional signed-package check with the original document and
the instance public key obtained separately. It extracts the source
again and recomputes the recorded code points and offsets. This
verification needs the project environment for text extraction.
It does not need any writing model weights.

```sh
cd backend
.venv/bin/python -B -m app.cli verify evidence.zip --public-key instance-key.pem --source-document original.docx
```

For a local automated check of source coordinates, run:

```sh
cd backend
.venv/bin/pytest tests/test_document_invisible_characters.py tests/test_billing_preflight.py
```
