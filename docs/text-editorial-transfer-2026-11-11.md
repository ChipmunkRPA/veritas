# Fresh editorial controls for passage evidence

The closed eLife intake provided 12 licensed, previously unscored science explainer excerpts. A separate frozen transfer scored all 12 unchanged originals, three deterministic copyedits, and ten paragraph substitutions taken from the same publisher article. The original cohort's development shortfall, 14 source exclusions and 11 unavailable editorial cases remain recorded. This transfer fitted no classifier or new threshold on these inputs. Exact source-group, text, attribution, editorial-span and unchanged-flank commitments were rechecked. The admitted work-level terms are CC BY 4.0. The earlier intake found no normalized 12-token overlaps with its historical inventory, and this comparison rechecked zero such overlaps with the 13 originals used to set the lexical point. Pretraining exposure is unmeasured. [Original source and admission record](text-elife-editorial-controls-v301-result.md)

## Fixed rules

The licensed four-class text checkpoint selected by the local configuration used the production sentence extraction and classification path to produce every new response. An audit bound each score to the actual token windows supplied to the model. The raw review baseline shows up to six contexts with responses of at least 0.5 in source order. Each context covers the selected sentence and its available neighbors. The strict rule uses the earlier development-only cutoff of 0.9961789846. Both are raw response points for this study. Article authorship confidence and the full mix of production checks are outside the comparison. [Earlier review-point comparison](writing-and-room-transfer-2026-11-11.md)

The model-free lexical cue compares each interior paragraph with both immediate neighbors. All three need at least 20 words. The features are moving-average type-token ratio with a 20-word window, log median sentence length, punctuation per word, and function words per word. The mean of absolute feature differences uses fixed scales of 0.08, 0.40, 0.08, and 0.10 respectively. The smaller distance to either neighbor is the paragraph response. The cutoff is strictly above the second-highest article maximum among 13 earlier development originals. The corroboration rule selects only raw-baseline contexts whose focus intersects a paragraph exceeding the lexical point. Displayed marks cover the actual scored context. A lexical shift also occurs in human editing, and a lexical point cannot establish authorship. [Mixed-authorship research](https://aclanthology.org/2026.findings-eacl.326/) · [Generation and editing stylometry study](https://aclanthology.org/2026.emnlp-main.1725/)

## Measured transfer

| Fresh wholly human arm | Raw review | Strict raw | Lexical cue | Raw + lexical |
| --- | ---: | ---: | ---: | ---: |
| Unchanged articles flagged | 12/12 | 4/12 | 2/12 | 0/12 |
| Copyedited articles flagged | 3/3 | 1/3 | 0/3 | 0/3 |
| Same-article paragraph substitutions flagged | 9/10 | 4/10 | 2/10 | 2/10 |

All 25 eligible files had full non-whitespace source coverage and at least one assessable interior lexical paragraph. The new CPU run used two threads, made 129 actual classifier forwards, and completed in 52.866 seconds after model load. Sampled peak process RSS was 2.29 GB. Neither a file flag nor marks overlapping an ordinary edit are evidence of AI use. Some files are variants of the same source, so the denominators do not represent 25 independent articles.

On a separate archived replay, 13 eLife confirmation originals and ten eligible mixed rewrites from two generators had exact native response records. Sixteen planned mixed cases remain unavailable. These sources were already used to test other rules and do not contribute new inference. Useful localization requires at least 0.30 non-whitespace character intersection over union with the known rewritten paragraph.

| Archived confirmation arm | Raw review | Strict raw | Lexical cue | Raw + lexical |
| --- | ---: | ---: | ---: | ---: |
| Unchanged articles flagged | 12/13 | 2/13 | 2/13 | 2/13 |
| Mixed rewrites with a useful location | 7/10 | 4/10 | 0/10 | 0/10 |

The frozen rule also ran on an archived federal-publication comparison with 30 source groups. The prior independent review verified all 262 available responses and their source positions. Raw review, strict raw, and raw + lexical flagged 43, 14, and 1 of 54 original views respectively. They localized 22, 7, and 1 of 54 long inserts and 1, 0, and 0 of 23 short inserts. The lexical cue alone had an assessable interior paragraph in 41 of 54 original views. Exact normalized 12-token matches with the recent explainer inputs remained zero. [Federal article provenance and limitations](text-gohumanize-transfer-v198-review.md)

## Fresh mixed passages

Two Apache-2.0 generator families each attempted one paragraph rewrite on every one of the 12 new source articles. Each admitted passage replaced a single interior paragraph and preserved the surrounding source text exactly. Qwen 2.5 1.5B used offline CPU bfloat16. SmolLM2 1.7B used offline Apple MLX 4-bit under a separate frozen decoder plan after the CPU run finished. The prompt, sampling settings, 200-token limit, end-of-sequence requirement, passage admission rules and classifier points were fixed before each run. [Generator asset review](evaluation-panels/text-elife-rewrites-freeze-v273.json)

| Generator | Eligible / attempted | Raw review | Strict raw | Lexical cue | Raw + lexical |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 2.5 1.5B | 5/12 | 5/5 | 2/5 | 0/5 | 0/5 |
| SmolLM2 1.7B | 1/12 | 1/1 | 1/1 | 0/1 | 0/1 |
| All eligible mixed files | 6/24 | 6/6 | 3/6 | 0/6 | 0/6 |

The four rule columns count useful localizations at the same 0.30 character IoU. The six eligible files came from five articles, with one article shared by both generators. Eighteen attempts were unavailable for scoring. Nine reached the token limit without ending, eight failed the length check, and one copied too much from its original. The pinned classifier made 28 forwards on the six eligible mixed files. The strict rule marked five files and misplaced the highlight entirely on one. Raw + lexical marked one and overlapped the rewrite below the useful IoU. No method or point was fitted using these results.

The raw review found every eligible fresh rewrite and also flagged every fresh original. The stricter point lost three useful locations and still flagged four of 12 fresh originals. The lexical gate reduced false flags and found no useful locations across the six fresh mixed files or ten archived mixed files. These results support keeping the current source-bound inspection path. The mixed panel's preparation exclusions and five represented articles leave the broader passage detection rate unmeasured. [Archived mixed cohort](text-elife-relative-paragraph-v292-result.md) · [Exact aggregate](evaluation-panels/text-editorial-transfer-2026-11-11.json)

The initial archived replay stopped at its legacy source-verification step because the current extraction module has a newer hash. The corrected readout revalidated the exact archived source, model-response, point and resource commitments and reproduced the known baseline and strict counts with zero model forwards. The initial fresh-generation invocation stopped before loading due to a shadowed local variable. The sandboxed MLX invocation stopped before loading because Metal access was unavailable in that runtime. Both completed runs retain their exact method and output commitments. Raw text and case-level responses stay local.
