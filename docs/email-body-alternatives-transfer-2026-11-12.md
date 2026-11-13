# Comparing plain-text and HTML body renditions in one email

Chipmunk Verify can optionally show material wording differences between the plain-text and HTML bodies of one supported `.eml` file. The ordinary evidence view, PDF and JSON use the same excerpts, MIME part identifiers and decoded Unicode positions. A wording difference can come from ordinary formatting or different versions of the message. It carries zero AI confidence weight and leaves the existing plain-text analysis intact.

## Scope

Set `VERITAS_DOCUMENT_EMAIL_ALTERNATIVES_REVIEW=true` to enable the check for new analyses. It defaults off. The input needs one unambiguous `multipart/alternative` group with exactly one plain-text body and one HTML sibling. The group can be the main part of a mixed message with separate attachments. The check decodes both MIME leaves and admits bounded, passive HTML with exact source mapping. It does not render HTML, load remote content or export active markup and destination URLs. It excludes terminal quotations, marked signatures, and narrowly identified unsubscribe footers. Interleaved quotations, ambiguous MIME, flowed text and unsupported or styled HTML remain unassessed. [MIME alternatives](https://www.rfc-editor.org/rfc/rfc2046.html#section-5.1.4) · [MIME main-view guidance](https://www.rfc-editor.org/rfc/rfc9787.html) · [Representation-gap study](https://ieeexplore.ieee.org/abstract/document/11714372/)

The comparison aligns words in the two mapped bodies, ignores case and punctuation changes and reports substantial differences next to shared text. It retains at most six differences and bounded text excerpts, with a count when further differences are omitted. Published positions refer to decoded parts and may be discontinuous across HTML tags or encoded character references. A signed package commits to both decoded part digests and their exact ranges. The offline source verifier independently rebuilds the comparison from the supplied original EML.

## Local evaluation

The source fixtures are synthetic, passive work messages. No private mailbox, sender-authenticity ground truth, or detector score entered the evaluation. The initial test used six development groups and 18 held groups. Positive locations needed the correct affected rendition, at least half the known changed words and at least 0.30 token intersection over union. Matched controls varied encoding, formatting, links, terminal quotes, signatures, reordering and attachments. Unsupported controls covered ambiguous or malformed MIME and HTML. Each transfer froze new subject groups before scoring. [Frozen commitments and source-range counts](evaluation-panels/email-body-alternatives-transfer-2026-11-12.json)

| Panel | Known wording differences located | Ordinary controls flagged | Unsupported controls correctly unassessed |
| --- | ---: | ---: | ---: |
| Initial held | 36/36 | 0/36 across 18 groups | 18/18 |
| Second transfer before correction | 12/12 | 3/12 | — |
| Same second transfer after correction | 12/12 | 0/12 | — |
| Fresh confirmation after correction | 12/12 | 0/12 | — |

The three initial false reports in the second transfer came from trailing service footers whose HTML link displayed only `preferences`. The correction requires a terminal opt-out instruction and a recognized link label with an HTTP or HTTPS destination. It preserved three substantive sentences discussing unsubscribe in the fresh confirmation, including a notice present only in the plain-text body. These controlled fixtures verify what the comparison reports. False-alarm rates for real mailbox exports remain unmeasured.

Four local Full analyses ran through simulated paid checkout. No unpaid event queued analysis. The same original with the option on and off produced the same plain-text source record, existing check results and overall decision. The new evidence matched across PDF, API, JSON and signed package. The offline verifier rejected an altered original and a shifted source position. Chrome showed the two exact source ranges in the ordinary view without external requests. [Local app receipt](evaluation-panels/email-body-alternatives-app-2026-11-12.json)

The public reader checks saved outcomes and decoded coordinate commitments without source messages. Repeating the native run requires the retained synthetic EMLs and exact construction scripts.

```sh
python3 -B -I -S backend/scripts/review_email_body_alternatives.py \
  docs/evaluation-panels/email-body-alternatives-transfer-2026-11-12.json
```
