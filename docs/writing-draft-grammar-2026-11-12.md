# Two-sentence writing checks on earlier drafts

Chipmunk Verify can now assess two adjacent changed sentences when an authenticated user supplies an earlier English `.txt` draft. Each version must have the same paragraph layout and an identical neighboring sentence on each side of the changed pair, with all four sentences in one paragraph. The check shows both exact source ranges and links to the underlying wording changes. It contributes zero authorship confidence. The two-check result is available in the browser, draft comparison PDF and JSON. The original single-sentence scope still applies to supported Word, PDF and email drafts.

## What the comparison found

The earlier-draft check holds the prior rule fixed. Two independently pinned local writing checks must both at least double their own AI-like response odds between two complete inputs. The check uses each changed pair and both unchanged neighbors. A useful location must cover at least 50% of the known edited non-whitespace characters and reach at least 30% intersection over union. The comparison keeps the first six positive native contexts from the existing passage check and the existing exact single-sentence paired result. [Earlier paired methodology and pinned classifiers](writing-paired-revision-replication-2026-11-11.md)

| Held panel | Exact wide edits located | Eligible genuine revised pairs flagged | Additional ordinary work groups flagged |
| --- | ---: | ---: | ---: |
| Initial matched excerpts | 2/3 | 0/7 | 0 |
| New native-paragraph replication | 1/4 | 0/3 | 0 |

The initial panel already located two of eight generated cases. The extension added two distinct work locations and brought that panel to four of eight. Two of the three exact-wide generated cases used Granite, and both were flagged. The OLMo case was missed. One Granite positive retained its original publisher paragraph. The other came from an exact excerpt flattened across publisher paragraphs. Among the initial seven genuine two-sentence changes, two had both source versions within one publisher paragraph.

The replication added one Granite location on a new work, at Unicode code points [88, 286) in its analyzed excerpt. Three eligible Granite cases yielded one flag and one eligible OLMo case yielded zero. A second OLMo completion qualified for the controlled generation study but preserved one of the two source sentences, so it was eligible only for the existing single-sentence path. The incumbent already located that completion. The combined useful count in the replication rose from one to two of five generated cases. The existing passage check flagged one genuine earlier control. The new rule flagged zero of three genuine later revisions and added zero ordinary work flags. These samples establish three added locations for this scope and leave false-alarm rates and transfer to other generators uncertain.

## Source selection and attrition

Fifteen dated, author-matched genuine works set the grammar points. Separate held sets contained 21 dated Wellcome and Gates journal works and 14 later-intake publisher works. The three panels have zero recorded author, work identifier or normalized 12-word overlap. Their source excerpts and model responses were already used for earlier, distinct comparisons. [Earlier source qualification](writing-journal-body-transfer-2026-11-11.md) · [Matched edit conditions](writing-source-conditioned-2026-11-12.md)

A separate two-batch Crossref intake froze 72 previously unused version-2 works. All 144 publisher XML files were acquired under a two-MB-per-file cap. Both versions needed explicit article-level CC BY 4.0, matching bylines, pre-2020 dated versions and exact earlier/later body text. Three works qualified. Fifty-two had no eligible native four-sentence context or overlapped known text, 12 overlapped known authors, three failed the date/byline rule and two failed metadata. The 48 frozen generator attempts used commercial-use Apache 2.0 checkpoints. Three Granite and two OLMo responses met the length, numerical, edit-distance and segment controls, with 43 attempts either unused or outside admission. Numeric preservation and exact source geometry were verified. Full factual equivalence, historical editing tools and model training exposure remain unverified. [Crossref metadata API](https://www.crossref.org/documentation/retrieve-metadata/rest-api/) · [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/) · [Granite model card](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct) · [OLMo model card](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct)

## Grammar cues tested alongside

The new grammar features compare the earlier and later changed span directly. One measures an increase in the share of closed-class words. The other measures changes to the sequence of grammatical word classes relative to lexical edit distance. Both use the existing English function-word categories and the second-highest genuine selection work as the operating point. Earlier research finds that lexical density can shift after AI copyediting, with weak separation from human writing. This evaluation uses that observation with matched genuine revisions and local features. [EMNLP study of AI editing](https://aclanthology.org/2026.emnlp-main.1725/)

| Check | Useful held locations | Ordinary work groups flagged |
| --- | ---: | ---: |
| Existing passage and single-sentence paired checks | 8/40 | 0/35 |
| Add closed-class share | 9/40 | 1/35 |
| Add grammatical sequence change | 8/40 | 0/35 |
| Add lexical edit distance | 10/40 | 0/35 |

The fixed lexical rule added two locations only in the earlier journal panel. Neither grammar cue beat lexical distance at a matched ordinary flag budget. The grammar and lexical authorship cues remain experimental.

## Reproduce the saved comparison

The standard-library reviewer recomputes genuine-only points, role and cohort denominators, useful locations, exact-wide eligibility, false-alarm groups and the lexical comparison from the text-free receipt. [Evaluation receipt](evaluation-panels/writing-draft-grammar-2026-11-12.json)

```sh
python3 -B -I -S backend/scripts/review_writing_draft_grammar.py \
  docs/evaluation-panels/writing-draft-grammar-2026-11-12.json
```

To reconstruct exact source inputs, retrieve the two version URLs per work and verify the XML SHA-256 values. Parse both JATS bodies using the [existing publisher reader](../backend/scripts/writing_journal_body_holdout.py). The receipt records normalized block hashes and version-specific positions. Preserve the flanks while inserting the first qualifying generated target. For the replication, use the exact stored prompt templates alternately, eight attempts per family per work in receipt source order and consecutive seeds from the family base plus eight times the zero-based work index. Require exactly two complete sentences, 12–70 words, within the larger of four words or 15% of the genuine target length, within 0.15 normalized word edit distance of the matched genuine change, and the same numeric multiset as the source. The receipt pins both checkpoint asset sets and runtime. Pass the exact joined native context strings through the [existing local CPU scoring helper](../backend/scripts/writing_paired_email_revision.py). The portable reviewer validates recorded arithmetic. A native replay independently reopened 34 selected XMLs in the initial later and replication cohorts and verified 123 source sentences, 125 grammar rows, 497 native incumbent context occurrences, 53 paired model input sets, 110 held cases and 11 replication cases. The previous journal-native reviewer separately replayed all 74 journal cases and 264 unique native inputs. Exact publisher and generated prose remain local.

Use `python -m scripts.review_document_revision_export comparison.json earlier.txt later.txt` from the backend environment to recheck the source hashes, exact ranges and saved response arithmetic against the two original files. Historical version-1 exports continue to replay against their original scope, including old out-of-scope decisions for two-sentence pairs.

## Local application verification

An isolated Full worker processed the replication positive and its genuine matched revision after two simulated payments. No unpaid request created an analysis job. Real Chrome opened each completed analysis, uploaded the earlier draft, selected the optional check, followed the wording links and downloaded both files. The browser showed a joint increase at [88, 286) for the generated edit and no joint increase at [88, 275) for the genuine revision. The PDF bytes and JSON contents matched the API response. Both JSON downloads passed exact source replay. The prior Full result digest stayed identical on both cases after the optional comparison. The two runs produced zero browser page errors.
