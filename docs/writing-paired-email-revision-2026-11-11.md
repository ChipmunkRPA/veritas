# Comparing an analyzed message with an earlier draft

Chipmunk Verify can compare an uploaded earlier draft with the retained original of a completed local analysis. Enable `VERITAS_DOCUMENT_REVISION_COMPARISON_ENABLED=true` in development, then sign in and open a completed `.eml`, `.txt` or `.docx` analysis. The comparison requires the same format on both sides and an active revision. A session or API key must have access to the retained analysis.

The comparison highlights changed words and punctuation side by side. Intervening unchanged text stays unmarked. For email, the selected plain-text MIME body, marked quotes and signatures use distinct evidence sections. The source excerpts preserve their decoded spelling and spacing. Each range counts Unicode code points, includes its start and excludes its end. The same ranges, excerpts and file hashes are available in the standalone JSON and PDF. The comparison leaves the analysis result and its confidence calculation alone. The earlier file is transient, and the comparison requires no new analysis or payment.

Word comparisons cover the decoded text of top-level body paragraphs in `word/document.xml`. Tables, headers, footers, text boxes, embedded content, formatting and tracked revision markup are outside the comparison. Global Word offsets count code points in the nonblank body paragraphs separated by two line breaks. Each excerpt also reports its paragraph-local range and one-based body paragraph number, which counts blank paragraphs. The JSON includes its zero-based position among all Word body children so a reviewer can find the exact XML paragraph. Word pagination depends on the renderer, so the comparison uses these source positions.

Each input is limited to 1 MiB, 30,000 decoded characters and 2,000 words and punctuation tokens. Email comparisons need a readable plain-text MIME body and contiguous role sections. The comparison covers textual differences between the supplied versions. Establishing who wrote a change or which version came first requires separate evidence.

The local API accepts `GET` and `POST` at `/api/v2/analyses/{aid}/document-revision-comparison`. The `GET` response returns whether comparison is available and the retained extension. The `POST` uses multipart form data with the file field `earlier` and the optional field `include_pdf` set to `true`. It returns `comparison` and, when requested and available, `pdf` as base64. The browser’s JSON download is the `comparison` object. Responses use `Cache-Control: no-store`.

Local integration verified that the ordinary EML analysis waited for simulated Stripe payment before queuing, then reused its completed result for the comparison. The earlier file did not enter the object store and the comparison queued no new analysis. The rendered PDF and exact JSON excerpts both preserved accented text and emoji on the local Mac. Other PDF environments depend on their font coverage.

## Separate evaluation of paired writing checks

A new set of 61 paired MIME fixtures used previously scored source material. The seven known partial AI rewrites came from two Apache-2.0 generators. Each was placed in the message body, in the body beside a marked human quote, and inside a marked quote beside an unchanged body. Controls included 14 workplace substitutions from genuine government-duty prose, ten genuine forum revisions, eight added human quotes and eight byte-distinct EML pairs with identical decoded text. The earlier version was supplied exactly in each pair. This is a source-exposed development comparison. [Prior workplace panel](writing-email-phrase-persistence-2026-11-11.md) · [Genuine forum revision sources](writing-shared-anchor-ranking-2026-11-11.md) · [Evaluation receipt](evaluation-panels/writing-paired-email-revision-2026-11-11.json)

The current classifier read the later text with its standard three-sentence windows. A model-free arm located changed text using a role-specific comparison. The experimental paired arm required identical surrounding body sentences across versions. It checked whether two distinct pinned classifiers each at least doubled their own AI-like odds from earlier to later. These changes describe model responses. The experiment used the earlier fixed geometry without fitting an operating point. Recent research reports that editing leaves different stylometric effects from generation. [AI editing research](https://aclanthology.org/2026.emnlp-main.1725/) · [Current checkpoint](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615) · [Independent checkpoint](https://huggingface.co/rasbt/ai-text-detector-modernbert/tree/a6efd4bf6ecd852f332b2569ae0f98cae13b7985)

| Outcome | Current context | Exact comparison | Both paired responses |
| --- | ---: | ---: | ---: |
| Useful location in plain AI-edited bodies | 0/7 | 6/7 | 2/7 |
| Useful location in AI-edited bodies beside a human quote | 0/7 | 6/7 | 2/7 |
| Useful location for AI text placed inside a marked quote | — | 6/7 | — |
| Human workplace body passages highlighted | 1/14 | 14 revisions located | 0/14 |
| Genuine human forum revisions highlighted | 0/10 | 10 revisions located | 0/10 |
| Body flagged when a quote alone was added or edited | 1/15 | 0/15 body changes | 0/15 |
| Body flagged on unchanged text with different MIME bytes | 1/8 | 0/8 body changes | 0/8 |

A useful location required at least 0.30 non-whitespace character intersection over union and 0.50 changed-character recall. The two added paired hits came from different source groups and generators. Both classifiers’ later raw responses stayed below 0.15 on those hits despite their relative increases. One of seven generated body pairs and four of ten forum revisions lacked matching context, so they had no paired statistical reading. The statistical arm remains in research. The local feature uses the exact comparison.

The run completed 57 CPU forwards over 223 model-input combinations. A separate replay checked 122 exact MIME files, 1,046 changed-token offsets, 229 current context references, 74 paired input references and 147 outcome and source-group counts. The [research runner](../backend/scripts/writing_paired_email_revision.py) accepts a frozen local directory of paired fixtures and both pinned snapshots.
