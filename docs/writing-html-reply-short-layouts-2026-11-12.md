# Second writing check on one- and four-paragraph HTML replies

The optional second writing check now runs on supported English HTML-only
replies with four body paragraphs, or one paragraph containing exactly
four sentences, followed by an explicit trailing blockquote. Each checked
sentence must be wholly within a verified body paragraph. The browser,
PDF and JSON retain the recorded passage positions. A signed report
can replay the decoded MIME ranges against the original email. This check
carries zero weight in the overall authorship confidence.

The frozen test re-encoded all 100 previously scored plain replies as
HTML-only EML files, with explicit paragraphs, an explicit trailing quote,
simple inline markup, and alternating base64 and quoted-printable delivery.
The app admitted every exact supported delivery. An independent MIME and
XML reader verified the extracted text, role, and exact source characters
behind 1,543 ranges. All 79 ordered second-check inputs matched their
plain-text versions. The transfer reused existing responses without
rerunning the models. All 12 negative controls stayed outside the second
check. Controls cover absent or interleaved blockquotes, styled markup,
unmarked quotations, malformed nesting, and tables.
[Plain-reply source and decision commitments](writing-email-body-scope-transfer-2026-11-12.md) ·
[HTML delivery replay](evaluation-panels/writing-html-reply-short-layouts-2026-11-12.json)

| Body paragraphs | HTML deliveries | Known edits localized | Human controls flagged |
| --- | ---: | ---: | ---: |
| One | 43 | 2/19 | 3/24 |
| Four | 57 | 12/25 | 10/32 |

Every known location and human flag matched the plain reply. The panel
reuses already measured text and constructed email layouts from 28 work
groups. These counts establish input and output preservation for these
layouts. Accuracy on real HTML mailbox language and unsupported markup
remains unmeasured.

## Local app verification

Four isolated runs exercised the paid Full worker and installed classifiers.
The one-paragraph generated reply showed one corroborated passage, the
four-paragraph generated reply showed two, and a genuine four-paragraph
reply showed zero. API, JSON and Standard PDF agreed and native replay
verified 52 decoded MIME source ranges. An interleaved control completed
without the second check. No unpaid event queued an analysis. The signed
report verified against the source, rejected a different email and
rejected an altered scope. A historical signed HTML report also verified
under its original version. Chrome opened both passage links in the
four-paragraph report with no page errors.
[Local app receipt](evaluation-panels/writing-html-reply-short-app-2026-11-12.json)

## Replay

Use the retained writing evaluation directory as `panels` and a new
private directory as `run`. The runner checks the earlier original file
hashes and replays the shipping scope on each exact HTML delivery.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
for phase in freeze measure review; do
  backend/.venv/bin/python -B -m scripts.writing_html_reply_short_layouts \
    "$phase" "$run" "$panels" "$run/result.json" || exit 1
done
```
