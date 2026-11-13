# Local vocabulary changes in mixed-origin writing

The combined vocabulary check recovered zero new AI rewrite locations in
two held sets. It added flags on one of ten genuine Kubernetes works and
four of fifteen genuine eLife works. The vocabulary-richness measure alone
found one additional eLife rewrite and added flags on 12 of 15 genuine
eLife works. The checks remain experimental.
[Public measurements and decisions](evaluation-panels/writing-local-lexical-transfer-2026-11-12.json)

## Frozen comparison

A November ACL study found that lexical richness helped classify complete
texts across several model and text domains. A separate study uses changes
in nearby scores to propose boundaries in mixed writing. This experiment
asks whether vocabulary statistics can locate one rewritten sentence amid
unchanged article prose. It implements its own measurements with no model
inference, added code dependencies or new weights.
[Cross-domain linguistic features](https://aclanthology.org/2026.findings-aacl.10/) ·
[Local change-point study](https://arxiv.org/abs/2609.33787)

The check compares an exact native sentence with its two immediate
neighbors. Each sentence must have 12–65 alphabetic-bearing lexical words
and six distinct content terms. At least 50 words must remain outside the
three sentences. Compute the expected fraction of distinct words when
drawing 12 words uniformly without replacement from each sentence. This
keeps the lexical sample size fixed. Measure how far the center sits
outside the two neighbors' range. Separately measure how many of each
sentence's distinct content terms are absent from the remainder of the
same document. The center must exceed both neighbors. The combined check
takes the smaller departure. Each rule selects a single focus only when
the highest response is unique and above its own fixed point.

The points come from the third-highest genuine maximum among 12 older
technical works, counting their unchanged originals and same-work ordinary
substitutions together. Later Kubernetes sources contribute 14 rewrites
from two generators and 17 genuine controls from 10 works. eLife
contributes 24 rewrites from two other generators and 30 genuine controls
from 15 works. Source groups stay disjoint, and all panels reuse previously
scored articles. A useful location must cover at least half the known
changed characters after excluding whitespace and reach 0.30 intersection
over union. Advancement requires two additional source groups across
both generators in each panel, with up to one additional genuine flag
group per panel. The prior model responses are retained as the baseline.
[Kubernetes source and model commitments](writing-multiblock-independent-2026-11-12.md) ·
[eLife source and model commitments](writing-elife-exact-sentence-confirmation-2026-11-12.md)

## Held results

| Rule | Kubernetes useful edits /14 | Kubernetes genuine groups flagged /10 | eLife useful edits /24 | eLife genuine groups flagged /15 |
| --- | ---: | ---: | ---: | ---: |
| Existing context check | 0 | 5 | 1 | 0 |
| Existing plus vocabulary richness | 0 | 10 | 2 | 12 |
| Existing plus unfamiliar terms | 0 | 8 | 1 | 9 |
| Existing plus both vocabulary cues | 0 | 6 | 1 | 4 |

Every held file had an eligible sentence. The exact edited sentence met
the lexical geometry in 13 of 14 technical rewrites and 17 of 24
scientific rewrites. The combined cue marked only wrong locations on
four rewrites in each panel. Its four flagged eLife genuine groups were
the same in untouched and substituted versions. The best post-hoc
combined ranking could locate two Kubernetes edits at three genuine
groups flagged and no eLife edits with up to four false groups. Held
points were used only to describe ranking. The study gives no basis to
use these signals as evidence of AI authorship.

## Replay

The public reader recalculates each score using exact rational arithmetic.
It reconstructs all three selection points, 123 case decisions and the
held false-group comparisons. It rejected a changed point, result, source
range and partition group. The optional native run reopens both previous
case ledgers, checks their published commitments and rebuilds all 1,324
focus measurements from the exact source texts. The public packet contains
counts and positions without text or vocabulary.

```sh
python3 -B -I -S backend/scripts/writing_local_lexical_transfer.py review \
  docs/evaluation-panels/writing-local-lexical-transfer-2026-11-12.json
python3 -B -I -S backend/scripts/writing_local_lexical_transfer.py review \
  docs/evaluation-panels/writing-local-lexical-transfer-2026-11-12.json \
  --native backend/data/independent-evaluation
```
