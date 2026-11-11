# Sentence ranking on genuine cooking forum revisions

Comparing two candidate sentences in the same outer context put 8 of 9 generated rewrites first, compared with 3 of 9 using the current context ranking. It also put 7 of 10 genuine human revisions first. Requiring both rankings to agree removed the only useful location from the existing matched check. The results support further study of review ordering and leave production selection and confidence unchanged. [Local result and anonymized case counts](evaluation-panels/writing-shared-anchor-ranking-2026-11-11.json)

## Frozen comparison

In an eligible four-sentence passage, the current ranking compares the scores for sentences 1–3 and 2–4. Their centers are sentences 2 and 3. The shared-anchor ranking compares sentences 1, 2 and 4 with sentences 1, 3 and 4. Both inputs keep the outer sentences in the same order, so the difference comes from which center sentence was inserted. The two rankings use the same pinned classifier. Changing context can alter the response to style, length or transitions. [Pinned Apache-2.0 checkpoint](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615)

The agreement rank produces a first choice only if both rankings have the same unique winner. The evidence comparison starts with the existing matched marks, then tests a one-location cap and an additional agreement requirement. Both preserve the existing review point. The ranking comparisons use no score threshold. Useful localization requires at least 0.30 non-whitespace character intersection over union with the exact changed sentence.

## Fresh sources

The fresh panel contains ten Seasoned Advice threads with distinct original authors and documented revisions from 2018 through 2021. Each pair has three identical sentences surrounding one revised interior sentence. Six revisions came from the original author and four from other editors. The genuine edits include spelling, punctuation, clarifications and an explanatory rewrite. Both revisions in each pair have an explicit CC BY-SA 4.0 license. Exact original revision bodies, attributions and edited text stay local. [Publisher license policy](https://cooking.stackexchange.com/help/licensing) · [Revision API](https://api.stackexchange.com/docs/revisions-by-ids) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

Every scored file is one plain-text English paragraph of 68–134 words and four sentences. Both interior sentences meet the 12-word minimum. An excerpt can join adjacent publisher paragraphs with a single space. The exact normalized 12-token overlap check found no matches across groups or with 45,370 previously retained text entries. Model pretraining exposure is unknown.

Two pinned Apache-2.0 generators attempted one rewrite per source. All 20 attempts reached the end of the response. Nine met the earlier exact replacement, length, sentence and copy rules, with four from Qwen and five from SmolLM. Six responses failed the word geometry rule, four copied too much, and one failed the single-sentence check. The source pairs, methods and generation rules were fixed before scoring. The previous NIST, USGS and Global Voices cases were used for development only. [Qwen terms](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE) · [SmolLM terms](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674)

## Results

| Ranking outcome | Current context | Shared outer sentences | Both agree |
| --- | ---: | ---: | ---: |
| Exact generated edit ranked first | 3/9 | 8/9 | 2/9 |
| Exact genuine revision ranked first | 6/10 | 7/10 | 3/10 |
| Generated cases with a first choice | 9/9 | 9/9 | 2/9 |
| Genuine revision cases with a first choice | 10/10 | 10/10 | 3/10 |

| Evidence outcome | Existing matched | Matched, one location | Matched, one agreed location |
| --- | ---: | ---: | ---: |
| Useful generated edit locations | 1/9 | 1/9 | 0/9 |
| Human files flagged | 4/20 | 4/20 | 2/20 |
| Human source groups with a flag | 3/10 | 3/10 | 2/10 |
| Exact genuine revisions flagged | 3/10 | 3/10 | 2/10 |

On the 63 previously consumed files, the current and shared-anchor rankings each put 15 of 21 generated edits first. Requiring agreement reduced human flags from 14 of 42 to 9 of 42 and useful generated locations from 9 of 21 to 7 of 21 relative to the one-location matched check.

The fresh run made 13 CPU forwards over 97 distinct complete inputs. An independent replay verified 116 exact context geometries, 80 publisher sentence positions, all 10 genuine revision pairs, nine exact generated replacements and 133 outcome counts. The fresh result covers one forum and two generator families. The strong localization result measures sensitivity to changed text, including genuine human edits.

The [local runner](../backend/scripts/writing_shared_anchor_ranking.py) takes a directory containing a source manifest, a protocol and a newline-delimited case file with exact fixture text, source offsets, and hashes. Run `python -m scripts.writing_shared_anchor_ranking freeze RUN_DIR` from `backend`, then `python -m scripts.writing_shared_anchor_ranking run RUN_DIR --snapshot LOCAL_MODEL`. The [authored tests](../backend/tests/test_writing_shared_anchor_ranking.py) check exact input geometry, agreement, ties, and the distinction between rankings and evidence marks.
