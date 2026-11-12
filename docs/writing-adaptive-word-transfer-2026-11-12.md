# Smoothing word scores for short AI edits

Adaptive smoothing found one of 44 known eight-word edits across two previously tested cohorts, the same edit located by the unsmoothed comparison. Fixed three-word smoothing also kept that location. Neither smoothing method added a useful location at its frozen point. The experiment remains outside the app. [Public scores and replay](evaluation-panels/writing-adaptive-word-transfer-2026-11-12.json)

## Comparison

A recent token-level study smooths nearby scores and selects the bandwidth using an adaptive compatibility rule. This trial applies the idea to the two source-aligned word-score vectors already recorded by the project. The native model inputs, candidate phrases and source coordinates stay fixed. The study reuses saved responses from the same pinned writing checkpoint and makes zero new model calls. It uses no code, data or model weights from the paper. [Adaptive smoothing study](https://arxiv.org/abs/2607.21458) · [Original technical article comparison](writing-phrase-context-stability-2026-11-12.md) · [Original PEP comparison](writing-pep-phrase-replication-2026-11-12.md)

The arms use no smoothing, a fixed triangular width of three words, and adaptive widths of one, three and seven words. Triangular averages stop at sentence boundaries. In each of the two neighboring contexts, choose the widest estimate whose interval overlaps those for every narrower estimate. The interval radii are half the empirical 75th-percentile absolute gap between the two contexts at each width. The 11 separate ordinary selection works supply these radii. The bands measure empirical disagreement and have no probability interpretation.

All three arms propose the same eight-word regions. The score is the lower result from the two contexts. Within each context, subtract the stronger four-word flank from the lower of the whole phrase mean and central-four mean. A unique maximum must strictly exceed the second-highest ordinary selection work maximum. A useful location needs at least 30% non-whitespace intersection over union and 50% changed-character recall. The frozen gate requires useful gains on two works across both generators within a cohort, no lost unsmoothed locations and at most one additional ordinary work flagged.

## Results

| Held outcome | Unsmoothed | Fixed three words | Adaptive |
| --- | ---: | ---: | ---: |
| Kubernetes inserts usefully located | 1/15 | 1/15 | 1/15 |
| Kubernetes ordinary work groups flagged | 0/12 | 0/12 | 0/12 |
| PEP inserts usefully located | 0/29 | 0/29 | 0/29 |
| PEP ordinary work groups flagged | 3/20 | 2/20 | 3/20 |
| Useful top regions before the point, both cohorts | 21/44 | 21/44 | 17/44 |

The adaptive rule selected seven words at every measured position in both contexts, across 10,536 choices. The selection-derived compatibility radii were approximately 1.64, 1.66 and 1.66 in native margin units. The uniform width reveals a limitation of using cross-context dispersion for compatibility on these excerpts. In the PEP cohort, the adaptive arm flagged three edited files but missed the known span in all three. Even at the same retrospective budget of two ordinary flagged groups, none of the three PEP arms located an edit. These held files had prior project exposure under different scoring rules.

## Replay

The portable arithmetic check verifies the three operating points and 324 held dispositions. It uses only anonymous source groups, scalar scores and overlap counts. For native replay, run the two phases with the retained original study folders and the backend Python environment. The native runner checks 132 selection and 604 held input bindings and writes a new result in a private directory.

```sh
python3 -B -IS backend/scripts/review_writing_adaptive_word_transfer.py \
  docs/evaluation-panels/writing-adaptive-word-transfer-2026-11-12.json
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
backend/.venv/bin/python -B -m scripts.writing_adaptive_word_transfer freeze "$run"
backend/.venv/bin/python -B -m scripts.writing_adaptive_word_transfer held "$run"
```
