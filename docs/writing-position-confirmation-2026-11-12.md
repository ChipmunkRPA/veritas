# Fixed writing position check on historical revisions

A follow-up test kept the writing position check and its operating point fixed. It found one additional exact Granite rewrite in a Copernicus abstract and flagged none of the eight ordinary work groups. It missed all four admitted TinyLlama rewrites. Combining it with the incumbent would have located three of 12 known edits with the same one ordinary work group flagged. The candidate remains outside the app. [Frozen confirmation and source receipt](evaluation-panels/writing-position-confirmation-2026-11-12.json)

## Historical sources and controls

The panel contains four fresh Python Enhancement Proposal work groups, two PeerJ works and two Copernicus works. Each supplies an original and a genuine pre-2020 revision with the same neighboring sentences. The PEP cases use exact adjacent Git versions of body prose with explicit public-domain or CC0 statements in both blobs. The journals use exact Crossref publisher-deposited abstracts with explicit CC BY terms on both dated records. The qualifier excludes previously scored works and checks normalized 12-word overlap with the earlier writing panels. The underlying source archives were retained from prior intake. The paper records and Git history establish the versions, with unknown authorship tools and model pretraining exposure. [Python PEP version history](https://peps.python.org/pep-0001/#what-is-a-pep) · [Publisher source qualification](writing-paired-revision-replication-2026-11-11.md) · [CC BY terms](https://creativecommons.org/licenses/by/4.0/)

The candidate measures the contextual head's center response minus the greater response at the two outside positions. The frozen point comes from the earlier PLOS and eLife genuine group maxima. No current or held response changed the point or the fitted head. This test covers a single short English paragraph with exactly three complete sentences. Each experimental positive highlights the exact center sentence. [Prior fitted head and source setup](writing-sentence-context-head-2026-11-12.md) · [Initial frozen position rule](writing-two-context-transfer-2026-11-12.md)

The two pinned Apache-2.0 local generators each attempted four one-sentence paraphrases per work. The first eligible completion preserves surrounding sentences and meets the fixed output, length and change constraints. All eight works supplied an eligible Granite rewrite. Four supplied an eligible TinyLlama rewrite, and none of the PeerJ works supplied one. The predeclared criterion required eight admitted edits per family, added locations from both families across distinct works, complete known-position coverage, and at most one additional ordinary false-flag group. [Granite terms](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct) · [TinyLlama terms](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

| Source | Known edits | Incumbent useful | Position check useful | Ordinary works flagged, incumbent / position |
| --- | ---: | ---: | ---: | ---: |
| Python PEPs | 6 | 2 | 0 | 0/4 · 0/4 |
| PeerJ | 2 | 0 | 0 | 1/2 · 0/2 |
| Copernicus | 4 | 0 | 1 | 0/2 · 0/2 |
| Total | 12 | 2 | 1 | 1/8 · 0/8 |

The additional location is a controlled single-sentence edit. The limited sample and generator imbalance leave transfer unresolved. Across the initial F1000Research trial and this confirmation, the frozen check added four locations from four distinct works and no ordinary flags. All four additions came from Granite. Two admitted OLMo edits and four admitted TinyLlama edits remained missed.

A next test can keep this same point and enlarge the second-family panel before scoring. One concrete route is an independent batch of pre-2020 Python PEP parent/child versions outside the 200-work legacy roster, paired with a frozen short rewrite prompt from a locally verified Apache-2.0 Qwen Instruct snapshot. Retain both exact version blobs, require explicit terms and an unchanged neighbor on each side, publish exclusion counts, and complete the specified second-family admission before opening detector responses. The eight already scored groups remain out of that holdout.

## Replay

The CPU run made 14 forwards on 84 source-bound inputs. The native audit reopened all 16 historical version records and confirmed the exact excerpt positions. The public receipt contains hashes, Unicode ranges, raw responses, and local checkpoint commitments. The model-free reviewer binds the unchanged point to the prior receipt, recalculates each decision, and checks the overlap and source counts. Source prose and generated text stay local.

```sh
python3 -B -I -S backend/scripts/review_writing_position_confirmation.py \
  docs/evaluation-panels/writing-position-confirmation-2026-11-12.json
```
