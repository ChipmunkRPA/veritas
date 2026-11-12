# Isolating AI-assisted responses in writing evidence

The current check located **4 of 32** known one-sentence rewrites and flagged **0 of 42** genuine journal versions. An experimental check using only the AI-edited and humanized responses located **1 of 32** and flagged **3 of 42**. Adjusting those responses for each article’s surrounding writing located **1 of 32** and flagged **2 of 42**. Both experimental checks lost useful locations and added zero. [Results and frozen comparison](evaluation-panels/writing-assisted-class-transfer-2026-11-12.json)

## Frozen method

The existing response adds three output classes for generated, AI-edited and humanized writing. The candidate adds only the last two classes. A separate candidate compares that score with the median assisted response within the same file on the log-odds scale. All arms read the exact same native sentence contexts with the same pinned checkpoint and tokenizer. The response applies to the whole context. The winning focus is also counted as a location diagnostic. [Checkpoint and prior local evaluation](text-upgrade-evaluation.md)

The baseline uses the first six contexts at its existing 0.50 raw review point. The candidates select the unique strongest context. A top-total comparator selects one context using the combined response. Each alternate operating point uses the same ordinary-group allowance as the baseline on the 16 fitting groups. Two of those groups had an ordinary flag. The points were written and verified before the held inference started. Useful localization requires at least 30% non-whitespace character intersection over union and 50% recall of the known edit. The release criterion required three added works across both generators, at most one lost work, and at most one additional ordinary group flagged.

The fitting material comprises 16 archived Kubernetes blog groups selected by a hash of each retained group ID. Their before and after views set the research points. The held material comprises 21 disjoint Wellcome and Gates research works with article-level CC BY 4.0 terms. Each held work includes two authentic published versions. Each of two offline Apache-2.0 generation routes supplied 16 admitted target-only sentence rewrites. Exact text outside the known target is shared with the corresponding ordinary document. The held inputs were previously scored in a study of version comparison. This test scores each version by itself and leaves paired draft information outside the ranker. No exact native input occurs in both partitions. Checkpoint training exposure is unknown. [Technical blog qualification](writing-archived-passage-transfer-2026-11-12.md) · [Journal and generation qualification](writing-journal-body-transfer-2026-11-11.md)

## Held comparison

| Outcome | Current | Top total | Assisted classes | Assisted classes relative to file |
| --- | ---: | ---: | ---: | ---: |
| Useful locations | 4/32 | 4/32 | 1/32 | 1/32 |
| Exact changed sentence selected as the context focus | 3/32 | 1/32 | 0/32 | 0/32 |
| Generated files flagged only at an insufficient location | 0/32 | 0/32 | 1/32 | 0/32 |
| Ordinary files flagged | 0/42 | 0/42 | 3/42 | 2/42 |
| Ordinary source groups flagged | 0/21 | 0/21 | 2/21 | 1/21 |

The assisted-only arm lost useful locations in two held groups and added flags to two ordinary groups. The relative arm also lost useful locations in two groups and added one ordinary group. The primary candidate missed the release criterion. The results describe localization on controlled English technical prose. These raw class responses carry no calibrated authorship confidence.

## Replay

The offline run measured 237 fitting and 264 held contexts in 126 CPU forwards. Every token count matched the earlier complete-input receipt. The largest difference from any retained combined response was 0.00000003. A second standard-library calculation reconstructed the three fitting points and all 528 case-by-arm decisions using the exact context positions. Its alternate log-odds computation treats floating differences within 1e-12 as equality at the frozen boundary.

From the repository root, use an empty private run directory and the existing pinned local writing snapshot. The verifier reopens the committed source packets and responses without another model call.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
backend/.venv/bin/python -B -m scripts.writing_assisted_class_transfer freeze "$run"
backend/.venv/bin/python -B -m scripts.writing_assisted_class_transfer fit "$run" --snapshot "$snapshot"
backend/.venv/bin/python -B -m scripts.writing_assisted_class_transfer points "$run"
backend/.venv/bin/python -B -m scripts.writing_assisted_class_transfer held "$run" --snapshot "$snapshot"
backend/.venv/bin/python -B -m scripts.writing_assisted_class_transfer verify "$run"
```
