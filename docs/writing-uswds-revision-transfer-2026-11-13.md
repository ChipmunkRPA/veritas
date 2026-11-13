# Writing localization on pre-2020 USWDS revisions

This test uses a new U.S. Web Design System source bank. We pinned the
publisher's December 19, 2019 repository tree, examined 36 dated path/commit
pairs and retained one revision per page. The six qualified pages span four
editorial commits. The selected Markdown and HTML prose contains an actual
wording change with identical surrounding sentences. The test rebuilds compact
TXT excerpts from those sentences and retains exact positions in each excerpt.
The publisher dedicates the repository to the public domain under CC0 1.0.
Across the retained OpAI-Bench, F1000 and eLife test records, the twelve new
genuine excerpts shared no exact text hash or case-folded 12-word sequence.
Checkpoint training overlap remains unknown. [Pinned publisher tree](https://github.com/uswds/uswds-site/tree/37f277bbdce08fbcf9bd463fa0c4b33516ceed61) · [Pinned source license](https://raw.githubusercontent.com/uswds/uswds-site/37f277bbdce08fbcf9bd463fa0c4b33516ceed61/LICENSE.md)

The same earlier excerpts also served as the basis for one local rewrite
per eligible family. We froze six pages and two attempts per generator
before inference. Five Granite and four OLMo rewrites across five pages
met the single-sentence, length and edit-distance rules. Each admitted
rewrite replaces one earlier sentence and preserves all other sentences.
Historical controls are attributable to the published editorial record.
Generated examples establish where text was replaced. We did not evaluate
whether every paraphrase preserved every fact. All nine retained the original
sentence's numeric strings. Both generation checkpoints
and both classifier checkpoints have Apache 2.0 terms. [Granite](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct/tree/707f574c62054322f6b5b04b6d075f0a8f05e0f0) · [OLMo](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct/tree/48d788eca847d4d7548f375ad03d3c9312f6139e) · [Primary classifier](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615) · [Second classifier](https://huggingface.co/rasbt/ai-text-detector-modernbert/tree/a6efd4bf6ecd852f332b2569ae0f98cae13b7985)

## Matched evidence

Two pinned classifiers each processed 84 distinct native three-sentence
inputs locally on CPU. The primary-only comparator marks up to the first
six positive windows and represents the primary local reading, without the
Full worker's other checks. One candidate keeps primary marks only where
an exact changed token intersects the input. A second requires both check
odds to at least double in the corresponding changed window. This candidate
requires an equal number of native sentence inputs across drafts. The
document comparison itself reuses the existing app's native draft diff.
A useful statistical proposal must cover at least half of the replaced
sentence and reach 0.30 IoU in non-whitespace source characters.

| Method | Untouched excerpts flagged | Genuine revised excerpts flagged | Useful locations in generated rewrites |
| --- | ---: | ---: | ---: |
| Primary three-sentence response | 2/6 | 1/6 | 4/9 |
| Primary response intersecting a changed token | 0/6 | 1/6 | 4/9 |
| Increase in both paired responses | 0/6 | 0/6 | 3/9 |
| Either primary response or both paired responses | 2/6 | 1/6 | 5/9 |

The paired response recovered one previously missed Granite replacement
in a guidance page. It missed two replacements the primary check found,
both on one page. Adding the paired response recovered one extra location
and left the later genuine flag count at one. The source is shared within each
page and several pages share a genuine edit commit, so the measured gain
needs replication before enabling a statistical change. The test did not
fit a new point or change confidence weights.

The model-free draft comparison found changed tokens in all six genuine
revisions and met the same overlap rule on eight of nine generated
replacements. The one miss had changed tokens inside the replacement, but
they covered less than half of it. This reports observable wording changes
from two versions. It carries no AI attribution.

## Reproduce

The [ledger](evaluation-panels/writing-uswds-revision-transfer-2026-11-13.json)
records source paths and commits, exact input hashes, raw responses,
character positions and aggregates. It contains no source prose or generated
excerpts. The standard-library reader recomputes the individual arms.
The freeze rerun reconstructed the same source roster and weight commitments
from all 57 retained source blobs. Both CPU classifiers were run a second
time, and the resulting ledger matched byte-for-byte. The reader rejected
a shifted source range, omitted original and altered outcome count.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.writing_uswds_revision_transfer review \
  docs/evaluation-panels/writing-uswds-revision-transfer-2026-11-13.json
```

For native scoring, use the [36-record source intake](evaluation-panels/writing-uswds-revision-intake-2026-11-13.json)
and the generator assets named in the ledger. Save the intake as
`candidates.json` in a new scratch directory. Obtain each public raw file
under its pinned path and commit. Store it under `native/` with the SHA-256
of the UTF-8 string `path@commit` as its filename. Save the pinned license
as `LICENSE.md` in the scratch directory. The generation commands
require the project's existing Apple Metal MLX environment and pinned local
model snapshots. The scoring step uses the project's Python environment and
both pinned classifier snapshots. No external inference call is needed.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.writing_uswds_revision_transfer freeze "$run" --generator "$generator"
for family in granite olmo; do
  PYTHONPATH="backend:$mlx_packages" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    backend/.venv/bin/python -B -m scripts.writing_uswds_revision_transfer generate "$run" --generator "$generator" --family "$family"
done
PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  backend/.venv/bin/python -B -m scripts.writing_uswds_revision_transfer score "$run" --first "$primary_snapshot" --second "$secondary_snapshot" --output "$run/result.json"
```
