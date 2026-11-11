# Targeted historical source follow-up

Replay v228 and all 31 private v233 request receipts offline. Require the exact published v233 aggregate and the locally sealed aggregate to match that replay. Bind each planned target to the relevant private v228 review keys. Keep target titles, revision IDs, keys, history and source text private. The aggregate records only counts and commitments. [v233 result](text-goce-source-evidence-v233-result.md) · [v233 aggregate](evaluation-panels/text-goce-source-evidence-complete-v233.json)

Use at most six serial read-only requests to the English Wikipedia Action API, with the same user agent, timeout, byte cap and stop behavior as v233.

1. Fetch the exact revision and parent from the one cached history summary that credits a page-to-page copy. Require the existing article, child timestamp, parent and summary bindings. Preserve both original main-slot bodies for a separate comparison of the copied material with the candidate.
2. Fetch the peer-review archive specifically named by the historical talk page for its own subject. Use one revision at or before the historical talk query cutoff. Record the archive as unavailable if no eligible visible revision is returned. References added after the historical talk snapshot are outside the target plan.
3. For each of the two article groups with overlapping candidate histories, anchor at the parent of the oldest verified row. Each pair of v233 trails must share 15 of its 16 rows. Fetch up to two further batches of eight and verify the chain against the known anchor. Report when the history reaches its cap or has a lineage gap.

This follow-up collects source material for later review. It makes zero rights decisions and runs zero models. A copy credit alone does not identify which wording moved. The separate content review must compare the exact parent and child, examine the relevance of any archive material, and keep unresolved wording pending. [MediaWiki revisions](https://en.wikipedia.org/wiki/Special:ApiHelp/query%2Brevisions) · [Wikimedia reuse terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content)

```sh
python3 -B backend/scripts/text_goce_source_followup_v235.py \
  /private/tmp/veritas-wikipedia-goce-v228 \
  /private/tmp/veritas-wikipedia-goce-rights-v233-attempt01 --preflight

python3 -B backend/scripts/text_goce_source_followup_v235.py \
  /private/tmp/veritas-wikipedia-goce-v228 \
  /private/tmp/veritas-wikipedia-goce-rights-v233-attempt01 \
  /private/tmp/veritas-wikipedia-goce-rights-v235-attempt01

python3 -B backend/scripts/text_goce_source_followup_v235.py \
  /private/tmp/veritas-wikipedia-goce-v228 \
  /private/tmp/veritas-wikipedia-goce-rights-v233-attempt01 \
  /private/tmp/veritas-wikipedia-goce-rights-v235-attempt01 --offline
```
