# Single-file writing localization on archived technical prose

The current writing check located 8 of 53 known one-sentence AI rewrites. The frozen exact-sentence ranker located 5 of 53 by itself. Using both located 10 of 53, with two additional useful work groups and two new false-flag work groups across the genuine controls. The two gains came from Qwen rewrites. SmolLM2 added zero gains. The supplemental rule missed the frozen requirement for three additional work groups with at least one gain per generator and stays experimental.

| Outcome | Current windows | Exact-sentence ranker | Both |
|---|---:|---:|---:|
| Useful localization on all generated rewrites | 8/53 | 5/53 | 10/53 |
| Useful localization on Qwen rewrites | 2/27 | 2/27 | 4/27 |
| Useful localization on SmolLM2 rewrites | 6/26 | 3/26 | 6/26 |
| Generated rewrites with a flag only at an incorrect location | 2/53 | 6/53 | 1/53 |
| Generated rewrites with no flag | 43/53 | 42/53 | 42/53 |
| Genuine source passages flagged | 5/36 | 5/36 | 6/36 |
| Genuine matching passages or later version flagged | 1/36 | 2/36 | 2/36 |
| Work groups with any genuine passage flagged | 6/36 | 7/36 | 8/36 |

## Sources and controls

An initial intake froze 1,191 historical file edits from the official Kubernetes and PostgreSQL archives. It sampled 164 Kubernetes blog edits, 650 Kubernetes documentation edits and 377 PostgreSQL documentation edits. One Kubernetes blog edit met the exact prose geometry and same-contributor Git attribution rules. The accepted pair has a changed center within both Git diff ranges, exact source offsets and before and after line attribution matching the recorded revision contributor. Both genuine versions received zero flags. One generated rewrite from this work was admitted and all three rules missed it.

The second intake froze 309 archived Kubernetes blog paths. It qualified 35 distinct bylined work groups for two disjoint, word-count-matched genuine passages per work. Each passage has five consecutive sentences with exact source offsets. The first supplied the test rewrite sentence and the second supplied another genuine control. These 35 groups represent authentic published prose and their paired passages come from separate locations in the same published work. The single historic revision appears separately in the results. A byline records public author credit at the work level.

The exact archived notices were verified and retained locally. Kubernetes applies CC BY 4.0 to the archived website, and the PostgreSQL license expressly permits use and modification of its documentation for any purpose subject to preserving the notice. The run screened source fragments against ten prior project packets and recorded zero twelve-word overlaps. [Kubernetes website license](https://raw.githubusercontent.com/kubernetes/website/main/LICENSE) · [PostgreSQL documentation license](https://www.postgresql.org/about/licence/)

## Frozen comparison

Each generator made one local attempt per work. Qwen admitted 27 of 36 and SmolLM2 admitted 26 of 36 under the same complete-sentence and geometry checks. Accepted edits replaced exactly the middle sentence and left the other four intact. A result needed at least 30% nonwhitespace character intersection over union and 50% recall of the edited sentence to count as a useful location. Qwen2.5-1.5B-Instruct and SmolLM2-1.7B-Instruct were pinned by revision and weight hash with Apache 2.0 terms. [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) · [SmolLM2 model card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)

All three rules shared the incumbent classifier's exact native window responses. The ranker used the uniquely highest eligible interior score and the unchanged operating point from the five-work PEP study. The highest-ranked interior sentence matched the edited center in 21 of 53 generated fixtures before applying the operating point. Both useful gains from the supplemental rule were independent work groups. It preserved the incumbent's useful locations and added two distinct false-flag groups across the two genuine controls. [Prior PEP evaluation](writing-pep-editorial-intake-2026-11-11.md)

The audit replayed 360 archived character and line-range witnesses, all 125 held fixtures, 625 context-to-span bindings, and all decisions from 518 distinct exact model inputs measured in 130 CPU forwards. It confirmed zero exact input overlap with cutoff fitting and rejected a modified outcome. The held source is English technical blog prose with a limited excerpt geometry and two scripted rewrite generators. Commercial terms and a historical archive support reuse and provenance. Exposure in model pretraining is unknown.

## Reproducing the evaluation

The [public receipt](evaluation-panels/writing-archived-passage-transfer-2026-11-12.json) contains archive and artifact hashes, source groups, per-file dispositions and frozen decisions. From the repository root, supply the two pinned local Git archives under `historical`, the retained overlap packets, the previous PEP operating point and calibration ledger, and the pinned offline checkpoints. Use a new local output folder for `run` and the pinned MLX Python environment for generation. The exact publisher notices, source text, generated sentences and probabilities stay local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for phase in freeze qualify verify; do
  backend/.venv/bin/python -B -m scripts.writing_archive_author_intake "$phase" "$historical" || exit 1
done
for phase in freeze qualify verify; do
  backend/.venv/bin/python -B -m scripts.writing_archive_prospective_intake "$phase" "$run" "$historical" || exit 1
done
backend/.venv/bin/python -B -m scripts.writing_archive_author_transfer freeze "$run" "$snapshot" || exit 1
for family in qwen smol2; do
  "$mlx_python" -B -m scripts.writing_archive_author_transfer generate "$run" "$snapshot" --family "$family" || exit 1
done
for phase in lock-inputs score report; do
  backend/.venv/bin/python -B -m scripts.writing_archive_author_transfer "$phase" "$run" "$snapshot" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_writing_archive_author_transfer "$run" "$historical"
```
