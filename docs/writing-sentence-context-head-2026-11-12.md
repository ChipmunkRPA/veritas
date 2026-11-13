# Localizing edits in three-sentence excerpts and longer writing

A fitted sentence check located eight additional edits in three-sentence excerpts from 25 held works. Three were OLMo rewrites. It also flagged nine ordinary work groups where the current rule flagged none. A separately fitted direct sentence check added one Granite location and no ordinary flags. Neither met the frozen criterion for adding the check to the app. [Portable response receipt](evaluation-panels/writing-sentence-context-head-2026-11-12.json)

## Method and sources

Recent work adapts a frozen text encoder to make separate sentence-level predictions within a short context and studies the effect of noisy local scores. This trial uses the installed Apache-2.0 writing encoder and fits two small, regularized linear heads of 1,025 parameters each. The first learns the feature change between an original sentence and an altered one, relative to the unchanged neighboring sentences. At inference it uses the submitted text alone and compares each sentence with its available immediate neighbors. The second learns directly from contextual sentence features. A third arm measures the existing token margins on the exact same sentence and uses no new weights. Each experimental arm selects one uniquely highest eligible sentence. The incumbent retains the first six three-sentence windows at its existing response point. [Machine-Generated Text Localization](https://arxiv.org/html/2402.11744) · [LA-CPD](https://arxiv.org/html/2609.33787v1) · [Pinned checkpoint terms](writing-checkpoint-agreement-2026-11-11.md#frozen-inputs-and-comparisons)

The fit used 12 Kubernetes archive works. The fixed point came from the second-highest genuine work maximum in a separate set of ten F1000Research originals and ordinary published revisions. PLOS and eLife supplied 25 other works with 41 known rewrites and 50 ordinary files, one original and one same-work human-prose substitution per work. The held articles use CC BY 4.0. Across 820 cross-panel source-group pairs, no normalized 12-word source fragment matched. The checks also verified that no complete model input was shared between the fit, point and held stages. These sources were used in earlier project experiments. The held works did not enter either head's fit or point selection. The earlier long views include full retained excerpts. The short views paste three sentences, with the known revision in the middle for every changed and matched control view. These short held views contain 38–141 words. [Kubernetes sources and license](writing-multiblock-independent-2026-11-12.md#sources-and-frozen-comparison) · [F1000Research sources](writing-source-aligned-transfer-2026-11-12.md) · [PLOS sources and license](writing-semantic-continuity-transfer-2026-11-12.md#sources-and-controls) · [eLife sources and license](writing-elife-exact-sentence-confirmation-2026-11-12.md#source-qualification)

A useful location covers at least half the edited non-whitespace characters and reaches 0.30 intersection over union. All 41 known edits had eligible exact sentence reads in both views. The gate requires two additional useful works across held publishers with at least one OLMo edit, preservation of current useful locations by union, and at most one extra ordinary work group per held publisher.

## Held outcomes

| View and check | Useful locations / 41 | Additional Granite / 23 | Additional OLMo / 18 | Ordinary groups flagged / 25 |
| --- | ---: | ---: | ---: | ---: |
| Three sentences, incumbent | 2 | 0 | 0 | 0 |
| Three sentences, paired contrast | 8 | 5 | 3 | 9 |
| Three sentences, direct head | 2 | 1 | 0 | 0 |
| Three sentences, exact token margin | 0 | 0 | 0 | 1 |
| Longer excerpt, incumbent | 3 | 0 | 0 | 0 |
| Longer excerpt, paired contrast | 6 | 3 | 2 | 16 |
| Longer excerpt, direct head | 1 | 0 | 0 | 12 |
| Longer excerpt, exact token margin | 0 | 0 | 0 | 6 |

Adding the paired check to the incumbent would recover 10 of 41 short-view edits and eight of 41 longer-view edits. Its short-view flags covered three PLOS and six eLife ordinary work groups. Four untouched files and six ordinary substitutions flagged, with overlap by work. For longer views, four PLOS and 12 eLife groups flagged. The direct check's one additional location had no increase in ordinary flags, while five other generated short views received a wrong location. The point did not transfer reliably across publishers. The app retains its current writing check.

## Local replay

The local CPU run made 199 native forwards over 1,189 unique bound context inputs. The private audit reconstructed 2,525 context occurrences and 120 known ranges across all 298 views, including both views of the same work. An initial fitting preflight stopped on a tokenizer token carrying a leading space. The corrected freeze assigns each token by its non-whitespace character coverage. It ran before any completed fit, point selection or held scoring. The receipt retains both freeze commitments.

The portable reviewer checks the point inputs, ordered non-whitespace coordinates, selected locations, and held outcomes from the recorded responses. The private workspace retains the exact source text, feature tensors, and fitted head. The public receipt contains hashes, score values, and positions.

```sh
python3 -B -I -S backend/scripts/review_writing_sentence_context_head.py \
  docs/evaluation-panels/writing-sentence-context-head-2026-11-12.json
```
