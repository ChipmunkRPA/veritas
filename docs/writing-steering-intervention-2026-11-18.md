# Can an activation change locate a short rewrite?

We tested whether a small change inside a frozen language encoder helps
locate AI rewrites in a single upload. It improved separation on the
training reports and found zero additional edit locations on both held
panels. The current writing check stays in service. [Steer-to-Detect, September 30 revision](https://arxiv.org/html/2605.12890v2)

## Method and source controls

The experiment used pinned Apache-2.0 MiniLM weights and one independently
fitted 384-component vector. Each measurement used the focal sentence and its
actual neighbors. Only tokens inside the focal sentence contributed to its
representation. Sentences needed 9–70 words, their three-sentence input had a
256-token limit without truncation, and a file needed five eligible candidates.
The intervention added the same vector to the focal tokens after encoder block
four. The two comparison arms used the same base weights and pooled the last
two layers. The vector fit took three fixed epochs and used an ordinary/rewrite
loss on 48 matched GAO passages from eight publisher editor-and-operation
combinations. Neither arm receives an earlier copy during scoring. This is a
sentence-level adaptation of the paper's intervention, with an independently
implemented paired loss, a bidirectional encoder and no borrowed detector
weights or guarantees. [Pinned encoder and license](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/c9745ed1d9f207416be6d2e6f8de32d1f16199bf) · [GAO training selection and reuse terms](writing-token-supervision-2026-11-17.md)

Each arm can select one sentence per file. It compares that sentence’s
response with the median response of other eligible sentences. We froze each
arm’s operating point on 109 ordinary source groups before assessment. Five
calibration groups exceeded the corresponding point under each arm. A correct
location covers at least half of the marked non-whitespace characters and
reaches 0.30 intersection over union. There were no shared normalized 12-word
fragments across the fit, calibration and two assessment panels in the private
source audit. The exact input and method digests are in the receipt. Exposure
to the public prose during base-model training is unknown.

The fresh panel used 11 previously unscored file histories from *The Turing
Way* Community, eight contributor groups and ten chapters. It contains 11
archived originals and their 11 documented human revisions from 2019–2021.
The archive is pinned to `e2394b4940f88c8e571dba6c496d0f7ea7a52d2c`. Earlier
project paths and shared passages were excluded. Two pinned local editors
attempted a single rewrite per work. Four of 22 attempts, two per editor,
met the previously defined single-sentence change, unchanged-neighbor and
number-preservation criteria. A generator label establishes that the editor
modified a sentence under this procedure. The four cases cannot establish
performance on naturally occurring AI assistance or every meaning-preserving
edit. [The Turing Way Community and CC BY terms](https://book.the-turing-way.org/) · [Prior archive and generator procedure](writing-turing-editorial-history-2026-11-18.md)

The retained comparison had 50 edited files with 68 marked locations from
three editors and 115 ordinary files. The ordinary files include untouched
prose, publisher-dated revisions and staged same-source human-language
substitutions. The project had already examined those materials in other
studies. The incumbent comparison uses its archived exact ranges for this
retained panel. [Prior retained source audit](writing-remote-semantic-transfer-2026-11-18.md)

## Held results

| Measurement | Unsteered | Steered |
| --- | ---: | ---: |
| Fresh edit locations | 0/4 | 0/4 |
| Fresh ordinary files flagged | 1/22 | 2/22 |
| Retained edit locations | 0/68 | 0/68 |
| Retained ordinary files flagged | 7/115 | 4/115 |
| Retained edited files flagged at the wrong location | 4/50 | 1/50 |

Both arms assessed all 191 files. Their geometry covered all four fresh
labels and 67 of 68 retained labels. Before applying an operating point, the
steered arm put the correct location first in three retained files. The
unsteered arm did so in five, and uniform choices from the same eligible
sentences would yield 6.26 on average. The incumbent located two retained
edits and flagged 23 ordinary files. Adding steered ranges held that result
at two edits and increased ordinary flags to 25. The fit’s improved internal
separation failed to transfer to useful locations. Neither arm was added to
Chipmunk Verify or used for confidence scoring.

## Offline review

The single-thread CPU run processed 4,134 scoring windows per arm on Apple
Silicon. A timed replay took 66.3 seconds elapsed and 65.2 seconds of user
CPU time for model load, fit and both arms of calibration and assessment.
This timing excludes source gathering and producing the test rewrites. The
repeat reproduced the fitted numerical record and every score receipt byte
for byte. The portable receipt gives pinned source, method and weight
commitments, anonymous groups, positions, scalar readings and counts. It
omits passages, generator responses and intermediate representations. The
reviewer reconstructs both operating points and both sets of held counts
from the scalar readings without installing models. [Numeric receipt](evaluation-panels/writing-steering-intervention-2026-11-18.json.gz)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_writing_steering \
  docs/evaluation-panels/writing-steering-intervention-2026-11-18.json.gz
```
