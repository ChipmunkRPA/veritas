# Second writing check on plain-text email replies

The optional second writing check now covers supported plain-text replies followed by a quotation or signature. The app uses the first contiguous mapped reply body. It revisits an existing primary context only when every sentence in the input belongs to that body. The source packet records the reply boundaries and the exact checked sentence positions. The analysis view, PDF and JSON identify the scope. The second check contributes zero weight to the overall assessment.

## Exact input comparison

The replay reused the 100 previously evaluated English excerpts, including 56 human controls and 44 known generated edits. Each excerpt was delivered with a trailing quotation, a signature, and both. The quoted material comes from a distinct source group in the pinned panel. The combined variant includes alternate HTML and a fixture attachment. The existing plain-text MIME extraction assigns quote and signature roles heuristically. [Original source and response commitments](writing-checkpoint-agreement-2026-11-11.md) · [Plain-text quotation and signature conventions](https://www.rfc-editor.org/rfc/rfc3676.html#section-4.3)

Each delivery preserved the exact body text and 583 of the 683 original context inputs across all 100 excerpts. The final body sentence in each delivery acquired a neighboring sentence from the trailing material. The reply check excluded those 100 contexts. It selected 79 exact inputs per delivery for review, 48 of which had a corroborating saved response.

| Outcome per delivery | Clean body comparison | With a trailing quote or signature |
| --- | ---: | ---: |
| Useful generated-edit locations | 14/44 | 14/44 |
| Human files with a corroborated highlight | 14/56 | 13/56 |
| Second-check contexts with quote or signature text | 0 | 0 |

The exact input replay uses previously measured responses and measures coverage after adding email structure. It supplies no new accuracy estimate for real mailbox language. Quote-only, interleaved quotation, and HTML-only controls remained outside this second-check scope in all 24 deliveries. Two quote-only deliveries contained the same source text and had identical file bytes. The ordinary document analysis still handles its broader extracted input separately.

## Local worker and exports

Five isolated signed test checkouts ran the actual CPU worker. A supported reply with both a quotation and a signature reproduced the same corroborated focus at **[372, 516)** as its clean counterpart, with identical input hashes, context coordinates, response and token count. The evidence API, JSON and PDF identified the mapped body and the original MIME part. Independent replay rejected an altered body boundary and a context that substituted quoted material. Quote-only, interleaved and HTML-only controls completed with the second check omitted. Unpaid test events queued no analysis.

The signed evidence ZIP verified against the original EML and the instance's pinned public key. Its reply scope matched the API and report. Verification rejected both an altered original EML and an altered scope in the signed package.

## Reproduce

Use the retained agreement evaluation directory as `panels` and a new local `run` directory. The native verifier independently reads the MIME body and recovers each selected model input from exact source positions. It verified 300 decoded bodies and 237 selected input occurrences. The public reader verifies the decision ledger without source prose.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze run; do
  backend/.venv/bin/python -B -m scripts.writing_email_body_scope_transfer "$phase" "$panels" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_writing_email_body_scope_transfer "$run" --panels "$panels"
python3 -B -I -S backend/scripts/review_writing_email_body_scope_transfer.py \
  docs/evaluation-panels/writing-email-body-scope-transfer-2026-11-12.json
```
