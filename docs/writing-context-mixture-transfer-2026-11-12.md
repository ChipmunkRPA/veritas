# Estimating sentence locations from overlapping writing checks

Estimating sentence scores from overlapping context responses found one narrow location at two false human source groups. The raw agreement ranking found two locations when evaluated on the full inputs it actually read. The new method stays experimental because its sharper locations came with reduced measured-context coverage. [Aggregate outcomes and commitments](evaluation-panels/writing-context-mixture-transfer-2026-11-12.json)

## Fixed method

LA-CPD uses local score consistency and sustained changes to segment a document. This experiment asks a different question about the overlap in our own evidence. A classifier input contains two or three sentences, and the model response applies to the input as a whole. Use the exact membership of every sentence to form a matrix of equal-weight context averages. Solve for sentence responses on the log-odds scale with ridge shrinkage to the excerpt mean and a smoothness penalty between adjacent sentences. Freeze both penalties before the run. The linear average is a working assumption about the two classifiers. It cannot establish what either classifier would say about an isolated sentence. [LA-CPD research](https://arxiv.org/abs/2609.33787) · [Implementation](../backend/scripts/writing_context_mixture_transfer.py)

Reuse 100 previously scored English excerpts, all 683 exact context occurrences and the pinned responses from both classifiers. The 57 grammar-seams files from 16 groups set operating points. The 43 option-order files from 12 disjoint groups supply 19 known edits and 24 human controls. The source banks, generator families and responses had prior project exposure. Each human source group includes an original and a human splice. The study ran without new model inference or downloads. [Earlier source and model qualification](writing-checkpoint-agreement-2026-11-11.md)

For each arm, select its strongest focus. Measure the same focus sentence and its existing complete input separately. Useful locations need at least 0.30 non-whitespace character IoU against the edit. At the frozen points, count false groups on the held panel without adjusting the points. The held equal-alarm comparison uses its own order statistics to examine ranking descriptively. It does not set a service cutoff. The comparison includes raw first and second responses, their minimum, both reconstructed scores and their minimum, and reconstructed agreement relative to the excerpt median.

## Held comparison

| Identical held false human groups | Raw agreement useful focus / full context | Reconstructed agreement useful focus / full context |
| ---: | ---: | ---: |
| 0 | 0 / 0 | 0 / 0 |
| 1 | 0 / 2 | 0 / 0 |
| 2 | 0 / 2 | 1 / 0 |
| 4 | 0 / 3 | 1 / 0 |
| 6 | 0 / 3 | 2 / 1 |

At the frozen points that each produced three held false groups, raw agreement located two edits through full measured contexts. Reconstructed agreement localized one through its chosen focus and none through full contexts. Before selecting a cutoff, the reconstructed agreement ranked 11 of 19 edits above both matched human controls, compared with 9 for raw agreement. The strongest focus alone overlapped three edits, up from zero, while the full-context overlap fell from 13 to 9. Removing the excerpt median reduced matched ordering to 4 of 19. The earlier multiple-window app agreement rule found two useful held contexts and flagged three genuine groups. These results support keeping the existing exact scored context in customer evidence. The reconstructed ranking has no confidence weight.

## Replay

The runner recalculates each context hash and source position. An independent numerical method solves the same regularized objective as an augmented least-squares problem and reproduces all 100 source rankings and false-group comparisons. The result file contains aggregate counts and commitments. The text and private responses stay in the local panel directories.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze run audit; do
  backend/.venv/bin/python -B -m scripts.writing_context_mixture_transfer "$phase" "$run" "$panels"
done
```
