# Source intake for the short-insert experiment

Start with the author’s pinned IteraTeR document-level `train.json`. Its exact bytes are 2,088,096 and its SHA-256 is `3f60e9989879b7a51b994edbe4536a5bd30556670f47b6677815bf0ed45c0acc`. The file is JSONL. Use only rows explicitly marked `news` and independently resolve their numeric document IDs on English Wikinews. The source paper describes collection of five recent published article revisions beginning in July 2021. Leave the publisher’s dev/test files and existing project holdouts outside this intake. [Pinned training release](https://huggingface.co/datasets/wanyu/IteraTeR_human_doc/tree/3b0bdabb090d04062ebc17e54ac889a64f5cb791) · [Source method](https://aclanthology.org/2022.acl-long.250/)

The publisher file contains 481 rows. The 169 explicit news rows cover 156 pages. A read-only local inventory found 98 constructible retouches across 62 pages before verifying their original archive lineage, rights or historical overlap. The proposed panel requires 48 distinct admitted pages after those checks and matching. An initial bounded archive probe encountered HTTP errors. A slower follow-up checked three eligible pages and found exact excerpt witnesses in both adjacent registered revisions on one page. These counts do not establish rights or the total admitted roster.

## Retouch and rights requirements

The intake aligns before and after text by whitespace-delimited words. Each candidate requires a bounded replacement linked to a publisher clarity, fluency or style annotation at its exact original coordinates. It adds common words on both sides of the edit until both target envelopes are 8–16 words and preserves at least 12 unchanged words per side outside the target. The exported offsets are half-open Unicode code-point ranges into the actual excerpt, with the excerpt’s original offset in each publisher revision. The private record carries both the edited subrange and the larger matched target. The real retouch is an all-human negative under this study’s documented provenance.

For each candidate page, collect official MediaWiki Action API responses in a private directory. Save exact JSON responses and keep their SHA-256 values. Use the public English Wikinews endpoint `https://en.wikinews.org/w/api.php`, `format=json` and `formatversion=2`. Retrieve candidate histories with `action=query`, `prop=revisions`, `pageids=<page>`, `rvstart=2021-07-31T23:59:59Z`, `rvlimit=8` and `rvprop=ids|timestamp|flags|tags|userid|sha1`. Save each as `history-<page>.json`. For each of the relevant revisions use `action=parse`, `oldid=<revision>` and `prop=text|revid|categories`, then save as `render-<revision>.json`. Make requests serially, reuse exact saved responses and stop if the publisher rejects or rate-limits requests. [Revision API](https://www.mediawiki.org/wiki/API:Revisions) · [Parse API](https://www.mediawiki.org/wiki/API:Parsing_wikitext) · [API etiquette](https://www.mediawiki.org/wiki/API:Etiquette)

The later revision must name the earlier revision as its parent, both must be visible and predate August 2021, and the later revision must have a registered editor and no known revert tags. The reviewer checks the edit history for ordinary editorial provenance and bots or reversions. Both excerpts must appear exactly once, allowing only whitespace normalization, in their respective official rendered prose. The paragraph parser removes citation markers, tables, navigation, and block quotes. Separately inspect the source paragraphs for inline quotes, imported wire text and other third-party material. Exclude unresolved rights or lineage. This yields archived editorial controls with an attributable source and date. It cannot establish the identity of every contributor or whether any tool assisted an older edit. [Wikinews copyright policy](https://en.wikinews.org/w/index.php?title=Wikinews:Copyright&oldid=4991371) · [CC BY 2.5 terms](https://creativecommons.org/licenses/by/2.5/)

Use a private JSON array for reviewed page/revision pairs. Each record uses `page_id`, `before_revision`, `after_revision`, `history_sha256`, `before_render_sha256`, `after_render_sha256`, `license` (`CC-BY-2.5`), `license_policy` (the pinned Wikinews policy URL linked above), and `story_component` (a SHA-256 of a locally assigned story group). Require positive evidence before setting `wikinews_staff_or_contributor_prose`, `third_party_material_excluded`, `ordinary_editorial_retouch`, and `not_known_bot_or_revert` to true. Preserve the page and revision IDs privately for attribution. Article titles, source text, review comments and account details stay out of the aggregate.

## Selection and commands

Compare source excerpts against a private index of case-folded, whitespace-normalized 12-word SHA-256 values from already consumed project writing inputs. The intake can check the supplied index and reports only the extent of that screen. Model pretraining overlap remains unknown. Use one document per reviewed story group and reject overlapping 12-word fragments between selected articles. Match paired original and revised target lengths within two words. Sort by a fixed hash of the page ID, pair greedily, and assign the first eight dyads to operating-point selection and the next 16 to evaluation. Within a dyad each article can supply both the host and outside-context donor. If there are fewer than 24 constructed dyads, report the shortfall. A later generator study must preserve these groups, use the untouched and reformatted controls, and seal the complete outcome criteria before using model results.

Run a read-only publisher inventory.

```sh
python3 -B backend/scripts/text_short_insert_intake_v203.py /private/tmp/veritas-iterater-train-v203.json
```

Run the private-evidence admission once the archive responses and reviewed inputs exist. Omitting `--private-roster` gives an aggregate only.

```sh
python3 -B backend/scripts/text_short_insert_intake_v203.py /private/tmp/veritas-iterater-train-v203.json \
  --official /private/tmp/wikinews-v203-official \
  --rights /private/tmp/wikinews-v203-rights.json \
  --exposure /private/tmp/wikinews-v203-exposures.json
```

The optional private roster writes only after full admission, with owner-only permissions and exclusive creation, under the ignored evaluation data directory or a system temporary directory. The aggregate includes no article identifiers or individual text hashes. This intake runs no models and establishes no detector accuracy or serving admission.
