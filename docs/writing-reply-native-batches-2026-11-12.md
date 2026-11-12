# Exact second-check batches for additional email reply layouts

All four tested layouts retained their useful same-source writing highlights and added zero human-text flags in this panel. The comparison covers supported English plain-text replies with two, three, five or six body paragraphs and a trailing quote or signature. [Aggregate and independent source review](evaluation-panels/writing-reply-native-batches-2026-11-12.json)

## Why measure the actual request

The earlier replay found that reformatting changed sentence boundaries in 12 deliveries from four cases in one source group. Saved classifier responses cannot be reused when the exact model input changes. Both existing commercially qualified classifiers measured nine changed inputs and 24 exact controls. The largest raw response difference on an exact control was 0.00000012 for the primary check and 0.01635 for the second check. No control changed sides at the fixed inspection point. The initial freeze required much closer score parity, so it failed before producing a layout decision. [Original layout audit](writing-reply-paragraph-transfer-2026-11-12.md) · [Classifier and source qualification](writing-checkpoint-agreement-2026-11-11.md)

The follow-up recomputes the second check for each original and reformatted reply using the exact source-ordered context sequence that the app sends. It combines only identical complete requests, preserving the usual microbatch and token cap. Primary scores are reused only for exact input hashes, supported by the separate parity test. Changed primary inputs use the new measurements. The check reopens native MIME text and tracks each original known edit through the new offsets. Each layout must retain at least 90% of useful same-source locations and add at most one human-text flag. All selected contexts must be measured completely and stay within the first contiguous reply body.

## Results

| Reply body paragraphs | Tested replies | Useful generated locations, source → reformatted | Human files highlighted, source → reformatted | New human highlights |
| --- | ---: | ---: | ---: | ---: |
| 2 | 100 | 14 → 14 | 13 → 13 | 0 |
| 3 | 100 | 14 → 14 | 13 → 13 | 0 |
| 5 | 55 | 12 → 12 | 10 → 10 | 0 |
| 6 | 53 | 12 → 12 | 10 → 10 | 0 |

There were 33 distinct ordered requests, including the empty request, and 75 second-check context inputs. The CPU stage made 35 model forwards. The independent reader reopened every delivered email and mapped the precise source contexts, score bindings, and outcome decisions. All 24 HTML-only and interleaved-quotation controls remained excluded. The 308 deliveries reuse 100 known-origin excerpts across 28 source groups. These results measure layout transfer. They do not estimate accuracy on independent workplace mail or authorship reliability beyond the measured passages.

## Local app verification

The same passage check now covers the four additional layouts in supported replies when the second writing check is enabled. It inspects up to six primary-positive contexts wholly inside the contiguous reply body. Each context stays within one paragraph. The scope excludes signatures, quoted messages, HTML-only replies, and replies with interleaved quotations. This evidence contributes zero to the overall authorship confidence.

Seven local analyses exercised the paid worker with simulated payment. Four known-generated layouts reproduced two corroborated passages each. The known-human control reproduced its previously observed corroborated passage. Both excluded controls stayed outside the check. The verified MIME offsets matched the PDF, API, and JSON. Chrome opened both passage links in the three-paragraph report. The signed bundle verified against the original email, rejected a different original and a changed scope, and remained compatible with an archived version 2 bundle. [Local verification receipt](evaluation-panels/writing-reply-layout-app-2026-11-12.json)

## Replay

Use the retained evaluation source root as `panels` and the earlier completed layout run as `prior`. The two checkpoint directories contain the exact pinned assets. Set `initial` and `native` to new local directories.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for phase in freeze primary second; do
  backend/.venv/bin/python -B -m scripts.writing_reply_layout_rescore \
    "$phase" "$initial" "$panels" "$prior" "$primary_weights" "$second_weights" || exit 1
done
for phase in freeze score report; do
  backend/.venv/bin/python -B -m scripts.writing_reply_native_batches \
    "$phase" "$native" "$panels" "$prior" "$initial" "$second_weights" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_writing_reply_layout_rescore \
  "$native" "$initial" "$prior" "$panels" "$receipt"
```
