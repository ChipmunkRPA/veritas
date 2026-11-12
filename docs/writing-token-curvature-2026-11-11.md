# Exact sentence predictability in travel-guide revisions

A conditional token-predictability check beat both paired human controls on three of five generated edits. The current context classifier did so on all five. Each ranked two exact edited sentences first, and they found different locations. The sample supports keeping the new cue in research. [Complete local evaluation](evaluation-panels/writing-token-curvature-2026-11-11.json)

## What the check measures

A causal language model predicts the next token from the text preceding it. For each token in the exact focus sentence, the runner records the observed token's log probability, the expected log probability over the model's vocabulary, and its variance. It sums the differences between observed and expected log probabilities and divides by the square root of the summed variances. Higher values indicate token choices that were more predictable relative to the model's conditional distribution. The comparison uses an independently implemented same-model version of the Fast-DetectGPT statistic with the existing Apache-2.0 base checkpoint quantized to 4 bits in local memory. It compares raw rankings with mean log probability from the same model and the current classifier's native context response. No new operating point or authorship probability is fitted. [Fast-DetectGPT paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6b8c6f846c3575e1d1ad496abea28826-Abstract-Conference.html) · [Causal checkpoint and license](https://huggingface.co/Qwen/Qwen2.5-1.5B)

## Sources and controls

Eight Wikivoyage travel guides from 2020–2021 supplied distinct passages. Four pairs contain real adjacent revisions by different editors with one changed interior sentence and three unchanged sentences in the scored excerpt. Four other pairs use a genuine sentence from elsewhere in the same guide as a controlled replacement. All source sentences map to exact positions in the saved revision bodies. The scored text excludes embedded media, templates and external quotations. The guides permit commercial reuse and adaptation under attribution and share-alike terms. The original and adapted prose stay local. [Publisher reuse policy](https://en.wikivoyage.org/wiki/Wikivoyage:How_to_re-use_Wikivoyage_guides) · [Revision API](https://www.mediawiki.org/wiki/API:Revisions) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

Every case is a single English paragraph with four consecutive sentences, 68–134 words and two eligible interior foci. The two foci have exact token coverage, and every target comparison within a source has an identical token prefix. Human controls shared no normalized exact 12-token fragments with other source groups or 45,399 known prior texts. Model pretraining exposure remains unknown.

Two pinned Apache-2.0 instruction checkpoints each attempted one rewrite per guide offline. All 16 generations ended normally. The frozen word-count, sentence and copy rules admitted five, with two from Qwen and three from SmolLM. Eight failed word geometry, two copied too much, and one failed the single-sentence rule. The five admitted edits span four guide groups. One has an actual revision as its human control and four use the constructed replacement. The causal scorer shares a model family with one generator, so their results are shown separately. [Qwen generation terms](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE) · [SmolLM terms](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674)

## Local results

| Paired ranking outcome | Current context | Mean log probability | Token curvature |
| --- | ---: | ---: | ---: |
| Generated target scores above both human controls | 5/5 | 0/5 | 3/5 |
| Qwen target scores above both human controls | 2/2 | 0/2 | 1/2 |
| SmolLM target scores above both human controls | 3/3 | 0/3 | 2/3 |
| Exact generated target ranks first in its own passage | 2/5 | 0/5 | 2/5 |
| Human edit scores above its original | 2/8 | 3/8 | 4/8 |
| Actual human revision scores above its original | 1/4 | 2/4 | 2/4 |

Both curvature localization hits came from the SmolLM family and were missed by current context. Curvature missed both current-context hits. The paired comparison showed more sensitivity to ordinary human edits than the existing classifier. The narrow study covers short travel-guide text and one causal scorer. The serving check and overall confidence remain unchanged.

The local run made 21 causal forwards on Metal and six CPU classifier forwards. An independent replay checked 52 publisher source positions, 42 scored foci, 1,337 exact scored tokens, four complete-vocabulary moment calculations and 132 paired outcomes. It reproduced the same target token prefix in all eight groups.

The [runner](../backend/scripts/writing_token_curvature.py) accepts a frozen local directory with a source roster, protocol, admitted case file, and generation receipts. Its `freeze` stage pins the exact snapshot, geometry and inputs. Run `baseline` with the classifier snapshot and `causal` with the pinned offline MLX environment, then `aggregate`. The [authored tests](../backend/tests/test_writing_token_curvature.py) cover distribution arithmetic and source-token boundaries.
