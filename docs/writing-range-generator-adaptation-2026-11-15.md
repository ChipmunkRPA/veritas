# Training word-range detection on GPT-5.4 rewrites

The matched-original range head missed every GPT-5.4 rewrite in the two
report evaluations. This test asks whether the same architecture can locate
more revisions when its training set includes examples from that generator.
The prior and adapted heads both read one excerpt at a time and use the same
pinned encoder, range proposals, matching loss and training schedule.
[Previous range study](writing-paired-ranges-2026-11-15.md)

The adaptation adds 108 report pairs. Sixty come from the earlier report
panel and 48 from the first range replication. Those pairs are training
examples only in this study. The 48 works in the second range replication
stay in evaluation. An attempted new report intake found only four eligible
works after applying the documented source and normalized phrase exclusions.
The report comparison therefore reuses the separated second replication.
The ordinary calibration uses 72 earlier report works isolated from both
training sets and evaluation. Each head gets its own points from those same
works. [Publisher methodology](https://arxiv.org/abs/2606.06481)
· [Dataset and reuse terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench#licensing-and-reuse)

The transfer checks use the report points on 24 news and 24 abstract pairs
from the same benchmark release. The source plan checks work identities and
normalized 12-word overlaps, then verifies one changed interior sentence
and four unchanged adjacent sentences. A separate audit against public
writing receipts identified 40 transfer works with prior project exposure.
The eight unlisted works have their own result row. The transfer genres
are absent from the adaptation examples. Both heads also score 32 held
local Qwen report pairs that are separate from the added training works.

## Results

A useful location covers at least half of the changed words and reaches
0.30 word intersection over union. The tighter and wider settings are
research comparisons on the same ordinary calibration roster.

| Outcome | Earlier head, tighter → wider | Adapted head, tighter → wider |
| --- | ---: | ---: |
| GPT-5.4 report rewrites located | 0 → 0 / 48 | 0 → 1 / 48 |
| Matched ordinary reports flagged | 3 → 8 / 48 | 3 → 8 / 48 |
| Local Qwen report rewrites located | 3 → 3 / 32 | 0 → 0 / 32 |
| Ordinary Qwen-panel controls flagged | 1 → 7 / 32 | 0 → 7 / 32 |
| GPT-5.4 news rewrites located | 1 → 2 / 24 | 1 → 1 / 24 |
| Ordinary news flagged | 9 → 14 / 24 | 11 → 19 / 24 |
| GPT-5.4 abstract rewrites located | 2 → 2 / 24 | 0 → 0 / 24 |
| Ordinary abstracts flagged | 11 → 15 / 24 | 12 → 16 / 24 |
| Rewrites located on unlisted transfer works | 0 → 0 / 8 | 0 → 0 / 8 |
| Ordinary controls flagged on unlisted transfer works | 5 → 5 / 8 | 5 → 6 / 8 |

The adapted head's one correct report location did not also trigger on the
corresponding sentence in the original. At the wider point a jointly
calibrated decision kept this location and flagged nine ordinary reports,
one more than either head alone. With the target sentence supplied, the
adapted head put a useful range first in four of the 42 assessed report
rewrites with changed words, the same count as the earlier head. The
adaptation loses useful Qwen and abstract locations and adds ordinary
flags outside reports. The adapted head stays in evaluation.

## Replay

The numeric receipt retains source hashes, assessed coverage, scores, native
positions, paired changes in outcomes, source exposure and the work-level
joint calibration. Exact source prose and learned weights stay local.
The portable reviewer checks that training, calibration, and evaluation
works do not overlap, then reproduces the decisions using standard Python.
Two native CPU runs reproduced the weights and readings. The numeric
review produced identical output on Python 3.12 and 3.14.
[Numeric receipt](evaluation-panels/writing-range-generator-adaptation-2026-11-15.json.gz)

```sh
export PYTHONPATH=backend
module=scripts.writing_range_generator_adaptation
python3 -B -S -m "$module" review docs/evaluation-panels/writing-range-generator-adaptation-2026-11-15.json.gz
```

The native run uses the report and genre shards pinned in the receipt and
the [earlier replay assets](writing-paired-ranges-2026-11-15.md#independent-replay).
The report source is `test-reports_gpt-5.4.csv`. The other publisher files
are `news_gpt-5.4.csv` and `abstracts_gpt-5.4.csv`. The initial range head
is the earlier paired `range.safetensors`.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
"$runner" -B -m "$module" freeze "$report_shard" "$report_roster" "$run/locked.json" \
  --news "$news_shard" --abstracts "$abstracts_shard"
"$runner" -B -m "$module" run "$report_shard" "$report_roster" "$run/fit" "$run/locked.json" \
  "$encoder" "$word_weights" "$prior_paired_head" --news "$news_shard" --abstracts "$abstracts_shard"
```
