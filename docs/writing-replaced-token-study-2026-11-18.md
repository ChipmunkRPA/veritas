# Can replaced-token scores locate a short rewrite?

A frozen token discriminator gave usable readings for all 16 known short rewrites. Neither direction of its score located a rewrite at its fixed operating point. High readings added six ordinary-file alerts and low readings added four. The trial reused exact local source excerpts and the same incumbent report-selected writing ranges as the preceding context study. [Source and editing audit](writing-native-context-study-2026-11-18.md)

## Method

The checkpoint was trained to distinguish original input tokens from plausible sampled replacements. It reads the observed word and both sides of its context in one pass. This tests a different objective from masked-token prediction or the earlier causal token losses. We used the original discriminator head and fitted no new weights. Recent detector research using a related pretraining objective reported that adding paraphrase augmentation and a separate contrastive loss did not improve its best configuration. Its document-level results do not establish the quality of this sentence locator. [ELECTRA paper](https://arxiv.org/abs/2003.10555) · [Recent DeBERTa-ConPara study](https://aclanthology.org/2026.aacl-long.82/)

Each candidate receives the exact preceding, focal, and following sentences of the submitted text. The limit is 256 tokens with no truncation. A focal sentence needs at least 10 alphabetic words with complete WordPiece offsets and coverage of at least 80% of its alphabetic words. The reading averages the logits of each word's pieces, trims the lowest and highest tenth of the eligible word readings, then averages the remainder. Each file needs at least five measured sentences. We center each sentence reading on the median of the other sentences and divide by their median absolute deviation multiplied by 1.4826, with a minimum scale of 0.15 logit units.

One direction ranks unusually high replacement scores. The other ranks unusually low scores to test the hypothesis that a rewrite can make the wording smoother. Each keeps its single strongest sentence and requires a strict excess over the second-highest maximum from 12 untouched calibration groups. Six calibration groups are historical prose and six are economic reports. None shares a source group with an assessment file. The selection rule and both operating points were saved before assessment inference. A location must cover at least half the labeled non-whitespace characters and reach 0.30 intersection over union.

## Results

Assessment used eight previously studied short rewrites plus one court and seven government-speech rewrites from the preceding study. The transcript controls include genuine source text, staged substitutions using genuine language, and naturally archived revisions. The two speech runs share eight ordinary inputs. After combining those duplicates, there are 16 labeled edits and 36 ordinary files. All 52 unique files had enough measured sentences, and every known edit had an eligible range. This is a retrospective comparison of a new signal on previously examined sources. [Exact prior cohort descriptions](writing-native-context-study-2026-11-18.md#calibration-and-control-roles)

| Measurement | High replacement score | Low replacement score |
| --- | ---: | ---: |
| Correctly localized edits out of 16 | 0 | 0 |
| Correct top range before applying the point | 1 | 3 |
| Ordinary files flagged out of 36 | 6 | 5 |
| Ordinary files newly flagged compared with the incumbent | 6 | 4 |
| Source groups with a newly flagged ordinary file | 4 | 4 |
| Edited files flagged at the wrong location | 2 | 3 |

The incumbent located zero edits and flagged one ordinary file. Neither candidate located any of the four rewrites without a prior meaning caveat, the three diagnosed meaning changes, or the rewrite that copied its successor. Before applying the point, the low direction ranked one of the four meaning-preserving edits first under the prior review and also ranked the copied-successor case first. The predeclared advancement rule required at least two additional meaning-preserving locations across two generators, at most two new ordinary source groups, and no more misplaced calls than correct additions. This trial did not meet that rule. The detector stays outside the app and adds no authorship-confidence weight.

The original training task concerns sampled token substitutions. A coherent rewrite can preserve local plausibility throughout a passage. The result covers a small English panel, and prior checkpoint exposure to the published source text is unknown.

## Runtime and replay

Inference ran offline on Apple Silicon using one CPU thread, macOS 26.5.2, Python 3.12.6, PyTorch 2.13.0, Transformers 5.15.1, and Tokenizers 0.22.2. It used float32 with the original Apache-2.0 checkpoint at revision `fa8239aadc095e9164941d05878b98afe9b953c3`. The checkpoint SHA-256 is `735154b1a8a78ba97da3c8c53c2e3d1d58b8dec5fa5caf0e144661c8bbe0e787`. It was loaded with tensor-only deserialization. The receipt includes checksums for all five model and tokenizer assets, frozen source and comparison inputs, and the local numerical results. [Pinned checkpoint and terms](https://huggingface.co/google/electra-small-discriminator/tree/fa8239aadc095e9164941d05878b98afe9b953c3)

The portable receipt uses anonymous aliases, half-open ranges, and final per-file comparison peaks. It omits source text, token or word readings, feature vectors, and local paths. The reviewer recounts calibration points, flags, and locations using the existing overlap rules. [Numeric receipt](evaluation-panels/writing-replaced-token-study-2026-11-18.json.gz)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_writing_replaced_tokens \
  docs/evaluation-panels/writing-replaced-token-study-2026-11-18.json.gz
```
