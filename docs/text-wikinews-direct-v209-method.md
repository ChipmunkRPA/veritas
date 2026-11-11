# Direct Wikinews retouch intake

This pilot derives retouch locations from adjacent official Wikinews revisions. The fixed sample draws at most four namespace-0 published pages from each of October 2018, April 2019 and October 2019. Use the official search index for each monthly category and `Published`, require complete results, exclude the earlier IteraTeR train article IDs, and rank the remaining page IDs by a fixed SHA-256 order. Fix all three selections before reading any article history. [Wikinews chronological archive](https://en.wikinews.org/wiki/Wikinews:Archives/Date/2019) · [MediaWiki category search](https://www.mediawiki.org/wiki/Help:CirrusSearch#Deepcategory)

Read at most 24 revisions per selected page with a strict pre-2020 cutoff. The later revision must name the earlier one as its parent and have a visible registered editor. Exclude known bot, revert, undo or rollback tags on either revision. Rank eligible pairs using possible short replacements in the raw revision bodies. The per-month render allowance is two pairs for the first two pages and one pair for the next two. The sequential archive transport permits at most 51 requests, respects the MediaWiki `maxlag` setting, and stops on a rejection or API warning. Exact replies are retained privately without overwrites. [Revision API](https://www.mediawiki.org/wiki/API:Revisions) · [API etiquette](https://www.mediawiki.org/wiki/API:Etiquette)

Check the later rendered revision’s archived `Published` category. Extract main-body paragraph prose from both renders and exclude candidate paragraphs bearing quotation marks or inline quote markup. The mechanical retouch must replace actual words, have 12 unchanged words on each side, and fit with its full flanks in one unquoted paragraph per revision. The unchanged padding gives both sides an 8–16-word target. Case, punctuation or number-only changes are excluded. Private records bind the before and after archive replies and preserve half-open Unicode offsets for the excerpt, target and replacement in each extracted revision. These are coordinates into the decoded paragraph text rather than HTML byte positions.

Separate source review is required before admission. Use the contemporaneous revision links and the pinned Wikinews copyright policy. Check the selected paragraphs for original licensed editorial prose, unlicensed or imported text, and quoted material. Review the edit history for an ordinary editorial retouch and exclude bots, reversions, and unclear cases. A site license and the age of the revision cannot independently prove who wrote the text. The history and publisher policy support source checks, and the exact page and revision IDs remain private for attribution. [Wikinews copyright policy](https://en.wikinews.org/w/index.php?title=Wikinews:Copyright&oldid=4991371) · [CC BY 2.5 terms](https://creativecommons.org/licenses/by/2.5/) · [Wikinews editorial review](https://en.wikinews.org/wiki/Wikinews:Reviewing_articles)

For an offline rights review, bind each decision to `page_id`, `before_revision`, `after_revision` and `window_sha256` from the private source record. The window digest is SHA-256 of UTF-8 JSON with sorted keys and ASCII escaping using the standard Python spacing. Use a specific rejection reason (`rights_unresolved`, `unlicensed`, `imported`, `quoted`, `bot_or_revert`, or `noneditorial`) or `approved` with the pinned policy, `CC-BY-2.5`, five positive source checks, and a private SHA-256 story group. The source checks are `licensed_original_prose`, `third_party_imports_excluded`, `quoted_material_excluded`, `ordinary_editorial_retouch`, and `bot_or_revert_excluded`. Admission excludes matching prior project 12-word fragments and duplicate articles, stories, or fragments. Unreviewed windows remain pending.

## Local result

The fixed sample completed all 12 pages in 49 requests. The three catalog pools contained 11, 6 and 31 eligible pages after project exclusions. The selected histories contained 268 mechanically eligible adjacent pairs, and 18 selected pairs were rendered. Of those 18 pairs:

| Disposition | Pairs |
| --- | ---: |
| Later archived revision lacked the `Published` marker | 11 |
| Published pair had identical rendered words | 4 |
| Published pair had no bounded replacement | 1 |
| Bounded change was nonlexical under the fixed check | 1 |
| Lexical replacement and flanks were inside one paragraph but failed the quotation screen | 1 |

The pilot produced zero mechanical windows. No source-level rights review or admission followed. This exact sample is closed. A lack of eligible windows in these selected pairs does not measure Wikinews retouch availability across the rest of the archive. The private cache contains raw archive replies and an empty candidate receipt. No models ran.

```sh
python3 -B backend/scripts/text_wikinews_direct_v209.py \
  /private/tmp/veritas-iterater-train-v203.json \
  /private/tmp/veritas-wikinews-direct-v209 --offline
```

The offline command verifies the private candidate receipt and prints aggregate counts. For a future nonempty sample, `--rights` and `--exposure` accept the private review records and a private array of case-folded, whitespace-normalized 12-word hashes from already consumed project inputs. The pilot publishes no source text or article, revision or editor identifiers.
