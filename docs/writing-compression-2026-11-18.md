# Locating sentence rewrites with lossless compression

Published research shows that whole AI-written texts can compress
differently from human-written text. The effect can weaken or reverse
in short passages. Recent multiple-author work also considered
compression across adjacent text as a possible boundary signal. This
experiment tests that idea at individual sentence locations in one
submitted excerpt. It uses the Python standard library for the
compression measurements. [Compression findings and scale limits](https://arxiv.org/html/2602.18152v1#S4) ·
[October 2026 boundary method](https://arxiv.org/html/2610.03565v1#S6.SS2)

Every eligible sentence receives three readings from raw DEFLATE byte
counts. The first ranks the most compressible text. The second compares
the sentence with sixteen deterministic rearrangements of its own
words. The third compares compression with the two preceding and two
following sentences as a dictionary against the same dictionary words
shuffled sixteen times. It ranks sentences whose compression uses
less of the original neighboring word order. The readings never see
an earlier draft. Normalization happens on the scored copy only.
Reported positions remain in the submitted excerpt.

## Frozen panel

Twenty-four independent published works supplied twelve calibration
groups and twelve held groups, equally split across NIST news, PLOS
One research and Python Enhancement Proposals. These passages were
screened against 91,046 distinct twelve-word windows from six earlier
source plans. The controls are untouched excerpts, two similarly
sized sentences exchanged within one source, and replacement by a
sentence of similar length from another work in the same source
family. The replacements use published wording. They are controlled
editorial changes. [NIST reuse policy](https://www.nist.gov/copyrights-disclaimers) ·
[PLOS article licenses](https://journals.plos.org/plosone/s/licenses-and-copyright) ·
[PEP copyright policy](https://peps.python.org/pep-0001/#what-belongs-in-a-successful-pep)

Two pinned Apache-2.0 checkpoints supplied exact interior sentence
rewrites. The admissible responses preserved the written numbers,
stayed within the length and edit-distance limits, and retained the
same sentence boundaries. Six generator–source pairs did not produce
an admissible result after three fixed attempts. The remaining
10 first-editor and 8 second-editor rewrites have exact character
ranges. These checks do not verify every fact in a rewritten
sentence. Earlier exposure during model training is unknown.
[First editor checkpoint](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306) ·
[Second editor checkpoint](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674)

Each experimental method can show one uniquely highest-ranked
sentence. Its operating point was set at the second-highest ordinary
source-group maximum from the 35 calibration files before the held
readings. The incumbent comparison uses the first six eligible
classifier cards and the configured document passage selector. It
leaves the overall analysis confidence and other writing checks
outside the comparison. A useful location must cover at least half
the changed span and reach 0.30 intersection over union after removing
whitespace. A combined reading adds only ranges that do not overlap
any selected incumbent range.

## Held results

| Reading | Known rewrites located | Untouched flagged | Within-source exchanges flagged | Other-article controls flagged |
| --- | ---: | ---: | ---: | ---: |
| Existing classifier and passage selector | 0/18 | 5/12 | 4/12 | 5/11 |
| Raw compression | 0/18 | 4/12 | 4/12 | 5/11 |
| Sentence word-order gain | 1/18 | 2/12 | 1/12 | 2/11 |
| Neighbor-conditioned compression | 0/18 | 1/12 | 0/12 | 1/11 |
| Existing plus sentence word-order gain | 1/18 | 6/12 | 4/12 | 6/11 |

The one correctly localized rewrite came from the second editor.
The same method highlighted the corresponding sentence in the untouched
original and the same position when supplied with a published sentence
from another article. This makes the successful location poor evidence
that the method identified AI editing. It also added one misplaced
range on another edited file. The neighbor-conditioned score's highest
unique rank matched four rewrites before applying its operating point,
against 2.58 expected from uniform selection among the same candidate
sentences. None of those four matches crossed its frozen point. The experiment leaves these
reading methods out of the app.

## Offline replay

The receipt includes article links, file hashes, measured scores, both
Unicode and whitespace-free character positions, and the incumbent
ranges. It excludes the source prose, model replies and model weights.
The offline reviewer reconstructs the operating points and the same
overlap arithmetic from the saved scores. [Numeric receipt](evaluation-panels/writing-compression-2026-11-18.json.gz)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_writing_compression \
  docs/evaluation-panels/writing-compression-2026-11-18.json.gz
```
