# Whole-document Content Credentials on PDFs

Chipmunk Verify can verify a supported signed PDF locally before it loads a
writing model. When a trusted signature covers the submitted PDF and the
signed record explicitly declares AI generation or editing, the result
shows that declaration and records that the writing checks were skipped.
It supplies no statistical confidence percentage or passage-level authorship
finding. The feature is opt-in with `VERITAS_DOCUMENT_PDF_CREDENTIALS=true`.
Payment is still required before the analysis job starts.

## What qualifies

The PDF catalog must directly associate the same uncompressed manifest store
as its embedded file entry. The active store read by the SDK must be the one
that the catalog names. The validator must find one active trusted manifest,
no validation failures, and a signed data hash covering the entire current
file except the manifest stream itself. The check reads the original bytes
offline with the bundled trust list. It makes no revocation or remote
manifest request. It accepts the tested provider layout, including its
placement of the media subtype on the file specification. The PDF appendix
permits additional layouts outside this check, including separate signature
exclusions and multiple stores across revisions.
[C2PA specification, PDF appendix](https://spec.c2pa.org/specifications/specifications/2.5/specs/ContentCredentials.html#_embedding_manifests_into_pdfs)
· [SDK support scope](https://github.com/contentauth/c2pa-rs/blob/main/docs/usage.md)

The local qualification covered one public OpenAI-signed PDF with visible
selectable text. The fixture is hosted by the upstream verification project
and stays outside this repository. The pinned bytes validated with C2PA SDK
0.90.14 against the bundled production trust snapshot. A PDF with added text
in a later revision, an unrelated PDF with the same attached manifest, a copy
without its catalog association, an authored unsigned PDF, and text containing
an unsigned generator name all continued the ordinary checks. This evaluates
one provider export and the scoped PDF layout. It does not establish provider
coverage for Word, general pasted text, or private watermarks.
[Upstream fixture and verification example](https://github.com/richardwooding/c2pa/tree/1119ea0c86b3ede12461b477b9069cefc3ee9e95#pdf)

## Independent local replay

Download the pinned original and use the backend environment. The receipt
contains the source and trust-list digests and verification outcomes without
quoting document contents.

```sh
scratch=$(mktemp -d)
url=https://raw.githubusercontent.com/richardwooding/c2pa/1119ea0c86b3ede12461b477b9069cefc3ee9e95/testdata/c2pa_chatgpt.pdf
curl --fail --silent --show-error "$url" -o "$scratch/provider.pdf"
export PYTHONPATH=backend
backend/.venv/bin/python -B -m scripts.review_pdf_content_credentials "$scratch/provider.pdf" \
  --expect-sha256 148974fecd2dca20e4e8050e4ee98d59ad0e2afbad7fa672dc9d9f5acffe284e
```

The [portable receipt](evaluation-panels/pdf-content-credentials-2026-11-13.json)
records the observed outcome. To check the signed and unsigned cases and the
simulated $9.99 checkout in Quick, Full and Deep, run from `backend` with
`CHIPMUNK_SIGNED_PDF_FIXTURE` set to the downloaded original.

```sh
CHIPMUNK_SIGNED_PDF_FIXTURE="$scratch/provider.pdf" \
  .venv/bin/python -B -m pytest tests/test_pdf_credentials.py
```
