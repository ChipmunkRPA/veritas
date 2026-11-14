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

## Look-alike letters

New analyses also report selected Greek and Cyrillic code points that can
look like basic Latin letters inside the same word. To qualify, the run
must contain at least three basic Latin letters, have fewer look-alike
letters than basic Latin letters, and contain only those two kinds of
letters. Complete non-Latin words, accented Latin spellings and runs with
combining marks are outside this rule. The check uses the 77 single-letter
ASCII mappings from the pinned Unicode 17.0 confusables data for Greek
and Cyrillic. It retains the original code point and names what the
character can resemble without substituting it in the analyzed text.
[Unicode Security Mechanisms](https://www.unicode.org/reports/tr39/#Mixed_Script_Confusables)
· [Pinned Unicode data](https://www.unicode.org/Public/17.0.0/security/confusables.txt)
· [Unicode License V3](../backend/app/evidence/data/unicode-license-v3.txt)

The pinned source SHA-256 is `091c7f82fc39ef208faf8f94d29c244de99254675e09de163160c810d13ef22a`.
The local validation checked all 77 mappings against that file and used
one mixed-script and one whole-script case for each mapping.

For HTML mail, the recorded source span covers the encoded entity.
A Word table cell retains its exact body path and paragraph position.
The report shows code points and plain-text names independent of font
rendering. Legitimate multilingual or technical text can contain these
patterns. They contribute zero weight to AI authorship.

The normal worker found one exact location in each of three authored
files after simulated USD 9.99 checkout. These covered plain text,
a Word table cell, and an HTML email entity. The unedited multilingual
control recorded no locations. PDF, API, JSON and signed replay agreed
on all four works.

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
