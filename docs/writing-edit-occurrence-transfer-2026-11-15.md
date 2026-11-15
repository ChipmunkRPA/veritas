# Distinguishing changed wording from AI rewriting

Two offline word checks tested whether examples of documented ordinary
revisions help distinguish AI rewrites. Both read each excerpt by itself.
An exact source comparison supplies changed-word labels during training and
evaluation. The experiment scores reproducible word differences in known
AI or ordinary revision pairs. Authorship of an individual word is outside
that measurement. [MixDetect](https://arxiv.org/abs/2609.32625) ·
[HACo-Det](https://arxiv.org/abs/2506.02959)

## Fit and evaluation

Both checks use the previous word study's frozen Apache-2.0 encoder. A small
head uses each word's contextual representation, its difference from its
sentence, and the sentence's difference from its neighbors. One check
directly predicts a word associated with an AI rewrite. The other predicts
a word change and then distinguishes associated AI and ordinary revisions
among the labeled changed words. It multiplies both responses for selection.
Both use the same training documents and the previous top-word selection.
[Prior word study and encoder terms](writing-word-localization-2026-11-14.md)

The plan selected 160 matched report works per training generator and 21
dated ordinary revision pairs from the previous journal study. Exact token
coverage left 52,369 useful interior words, including 1,639 labeled words
from generated rewrites and 44 from ordinary revisions in 14 distinct
works. This small ordinary edit sample limits the result. Each check sets
its two operating points using 72 separate ordinary report excerpts before
reading the held reports. The 93 held work pairs were used as held inputs in
the earlier word studies too. None of their groups entered these fits.
[Report and journal source roles and terms](writing-matched-encoder-2026-11-14.md)

## Wider operating point on the held reports

| Outcome | Prior word check | Direct candidate | Edit and origin candidate |
| --- | ---: | ---: | ---: |
| GPT-5.4 useful changed-word locations | 0/60 | 0/60 | 0/60 |
| Local Qwen useful changed-word locations | 4/32 | 4/32 | 4/32 |
| Ordinary reports flagged | 15/93 | 22/93 | 22/93 |

Three GPT-5.4 pairs lacked full word coverage. Each candidate selected
the exact edited sentence in four of the 60 cases. None of those word
highlights reached both localization criteria. A useful selection must
cover half the changed words and reach 30% word intersection over union.
The direct check gained two Qwen locations and lost two, with 17 newly
flagged ordinary files and 10 previously flagged ordinary files cleared.
The two-stage check gained and lost one Qwen location, flagged 14 new
ordinary files and cleared seven. The single held Qwen3 file had no
flag from either candidate. The tighter point also gave zero useful
GPT-5.4 locations. These results leave both candidates in evaluation.

## New genuine revision comparison

A separate intake froze 30 public eLife works using the pinned XML archive.
It checked the Git blob, publication dates, article identity, unchanged
author attribution, and the CC BY 4.0 license for both versions of record.
Five works failed version or archive metadata checks. Twenty-three had no
material body edit with two exact surrounding sentences on each side.
Two works qualified after excluding known project article IDs and checking
word-sequence overlap against the frozen report and journal corpus.
The initial v1/v2 scout found zero qualified pairs, mostly because the
first XML omitted the article body. The v2/v3 source roster was selected
before reading detector scores on these works. [Pinned publisher archive](https://github.com/elifesciences/elife-article-xml/tree/58e688ec3d4523eb1f3b03bed6bb440bf91e4786) ·
[Publisher reuse terms](https://elifesciences.org/terms)

All four eLife excerpts were measured at the fixed report points. The
direct and prior word checks flagged none. The two-stage check flagged
one earlier version and one later version at both points. This is a
small genuine transfer check. A larger sample of independently verified
editorial revisions is still required to estimate ordinary false alarms.

## Replay

The public record includes hashes, numeric word positions, operating
points and paired results. Source prose, feature vectors and model weights
stay local. A second fit and native transfer reproduced both receipts
byte for byte. The portable reviewer replays the counts without loading
a model or connecting to the network.

```sh
export PYTHONPATH=backend
panels=docs/evaluation-panels
primary="$panels/writing-edit-occurrence-2026-11-15.json.gz"
ordinary="$panels/writing-edit-occurrence-elife-2026-11-15.json"
module=scripts.writing_edit_occurrence_transfer
python3 -B -S -m "$module" review "$primary"
python3 -B -S -m "$module" review-ordinary "$ordinary" "$primary"
```

For native replay, use the exact report roster and pinned encoder and word
weights from the [previous native replay](writing-word-localization-2026-11-14.md#independent-replay).
The exposure history is JSON Lines in roster order with `text` and
`text_sha256` from `writing_word_localization.input_plan(roster)`. Produce
it with Python's default JSON separators, `ensure_ascii=False`, and one
line feed per row. The journal intake records its input hash. The fetched
sources can be replayed with `verify` using the same history file.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
module=scripts.writing_edit_occurrence_transfer
intake=scripts.writing_elife_version_body_holdout
"$runner" -B -m "$module" run "$roster" "$encoder" "$word_weights" "$run/fit"
mkdir "$run/elife"
"$runner" -B -m "$intake" freeze "$run/elife"
"$runner" -B -m "$intake" fetch "$run/elife"
"$runner" -B -m "$intake" prepare "$run/elife" --history "frozen-word-study=$history"
"$runner" -B -m "$intake" verify "$run/elife" --history "frozen-word-study=$history"
"$runner" -B -m "$module" ordinary-transfer "$run/elife" "$run/fit" \
  "$encoder" "$word_weights" "$run/replayed-ordinary.json"
```
