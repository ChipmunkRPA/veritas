# Passage evidence for HTML-only email replies

Chipmunk Verify now runs the second writing check on supported English HTML-only replies when that check is enabled. A reply can have two, three, five, or six body paragraphs followed by an explicit blockquote. Selected passages stay inside a single body paragraph. The original decoded HTML must map to exactly the same text the main analysis sees. Styled, ambiguous, interleaved, malformed, and unsupported layouts stay outside the second check. This evidence contributes zero weight to overall authorship confidence.

The [additional HTML reply layouts](writing-html-reply-short-layouts-2026-11-12.md)
now cover a four-paragraph reply and a one-paragraph reply with exactly four
sentences. That extension preserves the reply boundary and the same source
position requirements.

The report links to the passage in the extracted email text. The JSON includes positions in that text and in the decoded HTML message part. Both use Unicode codepoints. A signed evidence package can replay those positions against the original `.eml` file. Email attachments and sender authenticity are outside this check's scope.

## Local evaluation

The transfer replay used 308 supported HTML deliveries built from 100 previously measured excerpts in 28 source groups. The replay preserved all 2,079 eligible primary contexts and all 304 source-ordered second-check inputs, so it reused their saved responses. It reproduced 52 useful known-edit locations and all 46 existing human-text flags across repeated layouts. The independent reader reopened both MIME encodings, verified 6,915 mapped ranges and 852 context source ranges, and confirmed that all 36 unsupported controls stayed out. These are checks of source alignment and preservation of previous results. The repeated source deliveries cannot estimate accuracy on fresh email. [Transfer and independent audit](evaluation-panels/writing-html-reply-transfer-2026-11-12.json)

Three local worker runs used simulated payment and real classifier inference. Both supported replies reproduced the expected passages and exact MIME locations in the PDF, API, and JSON. The interleaved reply stayed out of the second check. Chrome opened both findings in the generated-example report. The signed bundle verified against the original HTML email, rejected a different original and a changed scope, and the verifier still accepted an archived plain-text bundle. [Local app verification](evaluation-panels/writing-html-reply-app-2026-11-12.json)

## Replay

Set `plain` to the retained plain-reply layout run and `previous` to the completed native-batch run. Set `run` and `receipt` to unused local paths. The scripts write and reopen the MIME deliveries locally. [Earlier plain-text run](writing-reply-native-batches-2026-11-12.md#replay)

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
for phase in freeze measure replay; do
  backend/.venv/bin/python -B -m scripts.writing_html_reply_transfer \
    "$phase" "$run" "$plain" "$previous" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_writing_html_reply_transfer \
  "$run" "$plain" "$previous" "$receipt"
```
