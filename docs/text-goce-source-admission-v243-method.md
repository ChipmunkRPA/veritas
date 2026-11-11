# Two source-reviewed historical copy-edit controls

The v228 pilot found seven exact copy-edit windows. The v233 and v235 evidence collections retained the original wikitext, historical talk and peer review, and source revision trails. Source inspection supports two windows from separate articles. The inspected text maps to original main-body prose on both sides, with a visible earlier introduction by a registered editor, exact parent and editor lineage, and contributor attribution paths. The inspection was assistant-assisted. External human reviewers remain zero. [v228 result](text-goce-intake-v228-result.md) · [v233 result](text-goce-source-evidence-v233-result.md) · [v235 method](text-goce-source-followup-v235-method.md)

Keep five witnesses pending. Three have earlier prose outside the observed body trail. One recorded retouch includes unchanged template expansion. The credited-copy witness has an original retouch target, while two words in its full surrounding context still trace to the credited revision. Preserve that full-window evidence and its pending status.

Replay the exact v228 roster and the published v233 and v235 aggregates with their private evidence chains. Bind the two reviewed windows to the historical revisions and contributor histories, original rendered text receipts, and exact source introduction receipts. Keep candidate titles, source and contributor links, history, revision and review identifiers, and item hashes inside the owner-only attribution and review records. The historical-revision policy and CC BY-SA 3.0 supply the applicable license basis for the selected original prose. [Historical revision policy](https://meta.wikimedia.org/wiki/Terms_of_use/Creative_Commons_4.0/FAQ/en) · [Wikimedia reuse terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content) · [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)

Rebuild the fixed exposure index from 20 known packets. These cover the historical writing focus, specialist, span, layout and rewrite panels, the early article and email manifests, federal generation journals, the GoHumanize source release, AITDNA span texts, the two earlier Wikinews archives, and the pinned IteraTeR training release. Validate the file, render inventory or input receipt digests before reading text. Keep each packet's unique 12-word hash count and combined commitment private. Text entries include related variants and repeated material. The overlap check covers the supplied packets, with model training overlap outside its scope.

Use the backend environment, which includes PyArrow. Preflight replays the evidence, verifies the exposure index and prints the two ready and five held counts without creating an admission cache. The admission command requires a new private directory. It passes the two approved source records and the full in-memory index to the existing v228 rights gate. The attribution, exposure and review sidecars are retained before the aggregate is sealed. Replay requires the exact sidecars and aggregate with zero requests. No mode runs a classifier.

```sh
backend/.venv/bin/python -B backend/scripts/text_goce_source_admission_v243.py \
  preflight /private/tmp /path/to/pinned/gohumanize/test.jsonl

backend/.venv/bin/python -B backend/scripts/text_goce_source_admission_v243.py \
  admit /private/tmp /path/to/pinned/gohumanize/test.jsonl \
  /private/tmp/veritas-wikipedia-goce-rights-v243-attempt01

backend/.venv/bin/python -B backend/scripts/text_goce_source_admission_v243.py \
  replay /private/tmp /path/to/pinned/gohumanize/test.jsonl \
  /private/tmp/veritas-wikipedia-goce-rights-v243-attempt01
```
