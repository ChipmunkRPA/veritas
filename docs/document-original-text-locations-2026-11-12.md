# Original text locations for writing evidence

After a document analysis, open **Media evidence**, choose a passage, and expand **Source anchors**. Each supported source anchor shows the exact original Unicode character ranges beside the extracted character range used by the writing check. The Standard PDF gives the same positions. The JSON export stores them under `evidence.coverage.original_text_locations` by passage and anchor.

For plain-text uploads, ranges refer to the original decoded file. For Word documents, they are relative to the original paragraph and shown with its body path. For email, they refer to the decoded selected MIME part. Mapped HTML may yield several ranges when an inline tag interrupts the source text. The gaps preserve those original positions. For PDF, the report shows extracted text offsets and the original page location, including its crop where one was recorded.

The app clips positions to the measured spans. Each range is zero-based and excludes its end. It omits native offsets for unsupported source mappings or spans without a complete map. The positions locate evidence from the existing writing checks.

The verifier reproduces these locations from the supplied original with `--source-document`. Local fixtures covered Unicode and UTF-16, Word table paragraphs, decoded MIME HTML entities and inline tags, and PDF page-only locations. The paid worker and Chrome test used an isolated local database and simulated Stripe events. Signed package verification and browser downloads checked the same stored evidence. [Local verification receipt](evaluation-panels/document-original-text-locations-app-2026-11-12.json)
