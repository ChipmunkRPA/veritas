# Padding repeatability on 26 other image sources

The prior image study found that requiring agreement under two padding
treatments reduced ordinary PNG highlights. This transfer asks whether
that effect survives new source groups and whether it helps beyond the
horizontal-flip check already used on the local app. Every treatment
uses the same installed checkpoint and source-coordinate readout. No
training or point selection uses this panel.
[Earlier padding study](image-edge-context-transfer-2026-11-14.md)
· [Horizontal-flip validation](image-regional-flip-confirmation-2026-11-13.md)

## Photo qualification

The panel uses ten LaMa removals, ten Stable Diffusion 1.5 insertions,
and six Kandinsky background edits from a pinned BG-REAL release.
The roster excludes known earlier padding and BG-REAL app-flip panel photos,
the published project exposure graph, the retained training and evaluation
photo groups, and verified exact or near-image matches. All 26 selected
photos passed intake. This establishes separation from those recorded
sources. Other local study exposure or pretraining overlap is unknown.
The originals carry recorded CC BY 2.0 attribution and the publisher's
contributions carry CC BY 4.0. The evaluation uses the edited files and mask
from the publisher and uses the existing permitted detector weights locally.
[Dataset license and attribution](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)
· [Pinned photo credit table](https://huggingface.co/datasets/anush-123/bg-real/blob/aaba6cd1ced4ce3c846be9641fed092adb4919d1/v2/meta/attribution.csv)
· [Benchmark method](https://arxiv.org/html/2607.26232v3)
· [Installed detector and permissions](native-regional-serving-v47-result.md)

Each photo contributes six PNGs at the original 768-pixel longest edge.
The panel retains the publisher edit for F5. The Kandinsky arm uses a
fixed native rectangle contained within the published edit mask. Each
source supplies sharp and feathered versions, the untouched original,
and matching local tone, blur and copy controls. Changed-pixel truth
is computed before any detector call. A useful location requires at
least 30% intersection over union and 50% of the exact changed pixels.

Compare the existing black-filled pass, a pass extending the closest
source pixel, a rule retaining original components with at least half
their selected cells supported in that second pass, the existing
horizontal-flip rule, and the intersection of the two repeat checks.
Both repeat views preserve the chosen original component's pixels.
They share the same weights and cannot supply independent confidence.

## Local results

| Single-upload check | Useful regions on edited files | Misplaced highlights on edited files | Ordinary highlights |
| --- | ---: | ---: | ---: |
| One original pass | 4/52 | 36/52 | 79/104 |
| Edge extension alone | 2/52 | 44/52 | 93/104 |
| Original regions also supported with extended edges | 4/52 | 34/52 | 73/104 |
| Existing horizontal-flip repeat | 6/52 | 26/52 | 50/104 |
| Require both repeats | 4/52 | 24/52 | 45/104 |

The existing flip check found four Stable Diffusion insertions and two
LaMa variants from one distinct photograph. Requiring both repeats lost
both useful LaMa regions, added no useful regions, and removed five
Kandinsky ordinary highlights across two photos. No arm localized a
Kandinsky edit usefully. The team is retaining the local flip setting and
keeping the padding check offline. This result measures region quality on
one benchmark publisher and three edit families.

The [earlier-copy WebP comparison](image-reference-webp-2026-11-14.md)
also used these sources with a supplied original. That paired result answers
where two files differ. It needs both copies and includes ordinary edits.

## Replay

The checked-in receipt contains the exact public source URLs as paths
under one pinned dataset revision, expected file hashes, publisher masks
and aggregate counts. It includes source metadata and measurements and
keeps the source images outside GitHub. A clean checkout can first review
the arithmetic with standard-library Python, then download only the
needed files and verify all inputs and changed-pixel masks without weights.
The clean-directory replay downloaded 84 publisher assets and reproduced
all 156 local upload and mask commitments. A separate arithmetic recount
agreed on every published count.
[Numeric and source receipt](evaluation-panels/image-edge-context-new-sources-2026-11-14.json)

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
receipt=docs/evaluation-panels/image-edge-context-new-sources-2026-11-14.json
python3 -B -S -m scripts.image_edge_context_new_sources review "$receipt"
backend/.venv/bin/python -B -m scripts.image_edge_context_new_sources hydrate "$photos" --receipt "$receipt"
backend/.venv/bin/python -B -m scripts.image_edge_context_new_sources verify-inputs "$photos" --receipt "$receipt"
```

The optional native measure step uses the same commercial-policy checkpoint
as the earlier local comparison. It records both image-preprocessing
commitments and the measured overlap counts. Run it in a newly hydrated
directory so it can write a separate result.

```sh
backend/.venv/bin/python -B -m scripts.image_edge_context_new_sources measure "$photos" --receipt "$receipt"
```
