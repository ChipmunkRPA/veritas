# Source evidence for the January 2018 copy-edit pilot

Replay the completed v228 archive offline and compare its private roster bytes before collecting evidence for its four candidate pages, six revision pairs, and seven windows. The roster fixes the ten unique article revisions and all window identities. This collection makes zero rights decisions and runs zero models. [v228 method](text-goce-intake-v228-method.md) · [v228 aggregate](evaluation-panels/text-goce-intake-complete-v228.json)

Use the English Wikipedia Action API and the v228 transport. The 33-request cap covers one batched attribution query, one exact revision query per page, up to two talk queries per page, two batches of at most 50 metadata revisions per page, and two batches of at most eight main-slot wikitext revisions per distinct parent. Requests are serial, with the existing user agent, one-second gap, 12-second timeout, 2 MB response cap, `maxlag=5`, no redirects, and no automatic retries. The first transport or response binding failure stops the collection and preserves the earlier receipts. [MediaWiki API etiquette](https://www.mediawiki.org/wiki/API:Etiquette)

- Use `action=query&prop=info&inprop=url|talkid&meta=siteinfo&siprop=rightsinfo` for canonical article and talk identifiers, attribution URLs, and the current site license. Keep the exact revision permalink and contributor-history pointer with each private review. [Page info](https://www.mediawiki.org/wiki/API:Info) · [Site info](https://www.mediawiki.org/wiki/API:Siteinfo)
- Use `action=query&prop=revisions&revids=...` for the exact historical revisions. Request the main slot, revision and parent IDs, timestamp, editor and tags, slot checksum, content model, and wikitext. Require the planned article and revision identities, chronological order, and the exact parent link. Record unavailable wikitext separately. [Revision API](https://en.wikipedia.org/wiki/Special:ApiHelp/query%2Brevisions)
- Fetch the last talk revision at or before the page's latest candidate edit and a supplemental snapshot at or before 11 November 2026 00:00 UTC, with one revision per request. Anchor article history at the latest candidate edit and each wikitext trail at its candidate parent. Replay only the official continuation within the stated limits. A broken parent chain or a limit reached with more history gets its own disposition. [API continuation](https://www.mediawiki.org/wiki/API:Continue)

Use a fresh owner-only destination for each collection. Each numbered receipt binds the exact query to its retained response hash or recorded transport stop. The aggregate commits to the verified v228 roster and ordered private receipts. Offline replay checks every response and the exact aggregate with zero requests. Keep private titles, revisions, editors, wikitext, windows and per-request receipts out of publication.

```sh
python3 -B backend/scripts/text_goce_source_evidence_v233.py \
  /private/tmp/veritas-wikipedia-goce-v228 \
  /private/tmp/veritas-wikipedia-goce-rights-v233-attempt01
python3 -B backend/scripts/text_goce_source_evidence_v233.py \
  /private/tmp/veritas-wikipedia-goce-v228 \
  /private/tmp/veritas-wikipedia-goce-rights-v233-attempt01 --offline
```

For later content review, locate both cached rendered windows in the exact revision wikitext. Review surrounding quotation, reference, template and table markup, then inspect edit comments, earlier visible text, and talk for quotations, copied or translated text, and third-party attribution or copyright notices. Ordinary citations can accompany original article prose. Hold a window when its source is ambiguous, relevant history is hidden, or available history cannot trace a suspect passage. Archive the historical source and contributor links, applicable license, and any modification notice with the private decision. Historical policy preserves CC BY-SA 3.0 for pre-upgrade revisions, subject to the applicable rights of the material actually reused. [Historical revision policy](https://meta.wikimedia.org/wiki/Terms_of_use/Creative_Commons_4.0/FAQ/en) · [Wikimedia reuse terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content)
