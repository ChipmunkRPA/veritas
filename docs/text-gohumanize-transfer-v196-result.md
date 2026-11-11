# Article test separates full-rewrite gains from short-insert misses

All six local batches completed. The two fixed text backbones made
1,952 forwards. All 262 constructed inputs were assessed by all four
methods. The 62 unavailable inputs were the short replacements and
their matching formatting controls that could not be constructed under
the fixed sentence rule. The corpus contains 54 excerpt pairs from
30 article groups. [Method](text-gohumanize-transfer-v196-method.md) · [Verified aggregate](evaluation-panels/text-gohumanize-transfer-complete-v196.json)

| Measure | Current passage check | Current plus canonical checks |
| --- | ---: | ---: |
| Full rewrites with a useful highlight | 2 / 54 | 28 / 54 |
| Longer AI replacements with a useful highlight | 1 / 54 | 1 / 54 |
| Short AI replacements with a useful highlight | 0 / 23 | 0 / 23 |
| Original passages with any highlight | 1 / 54 | 4 / 54 |
| Human characters highlighted on original passages | 267 | 3,931 |
| Human characters highlighted in longer replacement cases | 0 | 5,525 |
| Human characters highlighted in short replacement cases | 134 | 4,266 |

The full-rewrite gains concentrate in one generator. Useful highlights
rose from 2 to 27 across 39 GPT-4o mini rewrites and from zero to one
across 15 DeepSeek Chat rewrites. The combined method located 286 of
1,937 inserted AI characters in the short cases, alongside 4,266
human characters. It added coverage without producing a useful short
location under the frozen character-overlap measure. These findings
support further full-rewrite testing and leave the mixed-passage gap open.

The original passages and their formatting views share their article
lineage. The AI examples use two generation routes recorded by the
publisher, and the replacements follow the fixed sentence rule.
The measure is about source-aligned highlights. It supplies no verdict
for a customer document or performance estimate for email. Customer
confidence and serving settings were kept outside this study.
[Pinned source release](https://huggingface.co/datasets/gohumanize/gohumanize-open-humanizer-dataset/tree/cc066df21227f5fdb73b513c330181f946ce82ef)

A standard-library implementation independently rebuilt the source
coordinates and all four evidence masks across 324 planned cases. Its
coverage and metrics matched the complete aggregate exactly. The review
reused the committed model responses. [Independent method](text-gohumanize-transfer-v198-review.md) · [Review receipt](evaluation-panels/text-gohumanize-transfer-review-v198.json)
