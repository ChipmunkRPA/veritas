# Base and instruction model likelihood on edited article passages

The likelihood difference recovered **zero of 24** known one-sentence rewrites in held eLife articles and marked **five of 15** genuine work groups. The current context check found one useful location and marked no genuine groups. The experimental readings remain outside serving and the overall confidence calculation.

| Reading | Additional useful edit locations | Additional genuine groups flagged |
| --- | ---: | ---: |
| Base model alone | 1 | 5 |
| Instruction-tuned model alone | 2 | 4 |
| Instruction minus base | 0 | 5 |

All additional useful locations came from OLMo rewrites. The difference ranked the known edit first in three files before applying its operating point. Eight other generated files had a selected location that did not overlap the exact known edit enough. The criterion required two added work groups across both generator families and at most one added genuine group.

## Method

The 12 archived technical-blog selection works supplied 19 genuine article views, including seven ordinary substitutions. For each reading, the third-highest genuine group maximum set its internal operating point. A candidate required a unique best sentence strictly above that point. The held panel contained 54 files from 15 other works, with 24 admitted Granite or OLMo rewrites and 30 genuine original or ordinary substitution views. Source, generation and exact span commitments came from the earlier native article studies. The publisher groups and exact model inputs are disjoint across selection and held sets. The material had prior project exposure and model training exposure is unknown. [Archived selection articles](writing-multiblock-independent-2026-11-12.md) · [Held eLife sources and licenses](writing-elife-exact-sentence-confirmation-2026-11-12.md)

Both models received the same exact preceding sentence and target, joined with one space. The mean log probability uses only the native target's tokens. Complete nonwhitespace target coverage, matching token IDs and matching token positions were verified for every input. Each of the 449 unique inputs ran once per model on the local Mac, with a separate four-bit inference pass of about 31 seconds for each model. The useful-location rule required half the known changed characters and at least 0.30 nonwhitespace intersection over union. The study adapts the log-likelihood difference to a short source-bound target. It uses the already installed Apache-2.0 model pair, with no benchmark text, code or model files imported from the paper. [Implicit reward detection paper](https://arxiv.org/html/2604.21223v1) · [Base terms](https://huggingface.co/Qwen/Qwen2.5-1.5B) · [Instruction-tuned terms](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

## Replay

The portable reviewer rebuilds the three points from the selection groups, orders every held candidate, checks exact target coverage and recomputes the useful locations and false groups. The receipt includes hashes, Unicode ranges, overlap counts, token counts and saved likelihood values. Original text, prompts and token IDs stay local. [Public measurement receipt](evaluation-panels/writing-instruct-base-passage-transfer-2026-11-12.json)

```sh
python3 -B -I -S backend/scripts/review_writing_instruct_base_passage.py \
  docs/evaluation-panels/writing-instruct-base-passage-transfer-2026-11-12.json
```
