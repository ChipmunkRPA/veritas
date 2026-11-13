# Locating a changed sentence between two neighbors

A position-sensitive writing check found three exact AI rewrite locations missed by the current check on nine fresh F1000Research works. An independent left/right context comparison found two of those locations. Neither flagged an original or a genuine publisher revision. The new locations came from one generator, so the results support further testing. [Frozen measurements and source records](evaluation-panels/writing-two-context-transfer-2026-11-12.json)

## Why the earlier check flagged ordinary writing

The earlier fitted contextual check flagged 10 ordinary three-sentence files across nine works. Seven flags selected the first or last sentence, where the check had only one neighbor. All eight useful generated locations were in the middle. Restricting attention to the interior would still leave three ordinary flags, including two genuine prose substitutions and one unchanged original. Longer excerpts showed a different pattern. Twenty-one of 29 ordinary flags occurred away from the first and last sentences. The earlier check appears sensitive to asymmetric context in short excerpts and to other source differences in longer text. [Earlier study](writing-sentence-context-head-2026-11-12.md)

## Frozen comparison

The first candidate requires an exact interior source sentence and reads it independently with the preceding sentence and the following sentence. It retains the smaller of the two responses from the existing contextual head. The second candidate uses the original three-sentence response at the center and subtracts the stronger response from the two outer positions. The test also measures a center-only restriction at a separately selected point. A rank-margin variant of the left/right check gives identical results here because three-sentence files have one possible center. The test uses the existing locally fitted head and Apache-2.0 encoder. It borrows the ideas of comparable local windows, source-specific context and length-aware evaluation from recent work. [Local change-point research](https://arxiv.org/html/2609.33787v1) · [Length-conditional calibration](https://arxiv.org/html/2611.10901v1) · [Pinned checkpoint terms](writing-checkpoint-agreement-2026-11-11.md#frozen-inputs-and-comparisons)

Twenty-five earlier PLOS and eLife work groups supplied 50 unchanged and ordinary edited files. The second-highest ordinary work maximum set each internal operating point before the fresh scores ran. The study compares like-length excerpts. It does not establish exchangeability across publishers or support genre-specific customer calibration.

The fresh sample reopened 98 retained publisher version pairs and excluded previously scored work groups and exact 12-word overlap with retained writing sets. Nine source works qualified with dated v1/v2 releases, equal bylines, explicit CC BY 4.0 terms, and one substantively changed sentence with exact adjacent text. Publication dates range from 2015 to 2017. Each work supplies both the earlier original and the actual published later revision. The sources were retained for prior intake, and these nine works had no previous detector scoring in the reviewed project sets. [Publisher source method](writing-reference-revision-transfer-2026-11-11.md) · [F1000Research policies](https://f1000research.com/about) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Two locally installed generator families made three attempts per work. The first candidate meeting the frozen completion, one-sentence, length, copy and unchanged-neighbor criteria was selected without detector readings. Nine Granite and two OLMo edits qualified. The remaining OLMo attempts mostly failed the source-admission rules. The useful-location rule requires at least half of edited non-whitespace characters and 0.30 intersection over union. The advancement criterion requires two additional distinct source works across both generator families and at most one new ordinary work group. [Granite license](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct) · [OLMo license](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct)

## Fresh results

| Check | Granite edits found / 9 | OLMo edits found / 2 | Added useful locations | Originals flagged / 9 | Publisher revisions flagged / 9 | Ordinary groups flagged / 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current context check | 2 | 0 | 0 | 2 | 2 | 2 |
| Prior contextual head | 1 | 0 | 1 | 2 | 2 | 2 |
| Center-only restriction | 1 | 0 | 1 | 0 | 0 | 0 |
| Stable left and right contexts | 2 | 0 | 2 | 0 | 0 | 0 |
| Interior-versus-edge contrast | 3 | 0 | 3 | 0 | 0 | 0 |

The position contrast added exact locations in three separate works. Keeping those alongside the incumbent would give five useful locations in 11 known edits and the same two flagged ordinary groups. The two OLMo misses leave family transfer unresolved. This test had limited source diversity, with one fresh publisher and only two admitted OLMo examples. The candidates remain outside serving.

## Replay

CPU scoring completed 67 forwards on 395 unique bound inputs. The native audit verified 18 publisher XML sources and the exact original and revised excerpts, including each JATS source block and center location. Every tested generated focus received an eligible native read. The public receipt contains offsets, saved readings, source and input hashes, and generator commitments. Generated sentences and original prose stay local.

```sh
python3 -B -I -S backend/scripts/review_writing_two_context_transfer.py \
  docs/evaluation-panels/writing-two-context-transfer-2026-11-12.json
```
