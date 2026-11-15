# Test local word predictability against known AI rewrites

This comparison asks whether a single language-model pass can locate
changed words inside otherwise ordinary writing. The scorer measures
how likely each observed token was relative to the distribution of
possible next tokens. It uses the pinned Apache-2.0 Qwen2.5-1.5B base
checkpoint and reads only the submitted excerpt. Each verified word
position retains the summed difference between observed and expected
log probability and the summed conditional variance.
[Pinned checkpoint and license](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/8faed761d45a263340a0528343f099c05c9a4323)
· [Conditional probability curvature](https://arxiv.org/abs/2310.05130)
· [Token-level localization research](https://arxiv.org/html/2607.21458v1#S3)

## Fixed comparison

The absolute check scores every contiguous range of 3, 5, 8, 12, 18,
or 24 words inside each of three interior sentences. It also considers
complete sentences of 3–36 words. The second check measures the smaller
standardized difference from the immediately preceding and following
words, using three to twelve words on either side. The third check takes
the smaller score from these two measurements. Each selects at most one
range. It declines equal maxima and excludes a word if its pieces overlap
another word, lack complete source coverage, or include the leading
unpredicted token. The standardized scores rank possible ranges. They
carry no authorship-confidence interpretation.

The 144 separate ordinary calibration works supplied the points by genre
before held inference. They included 72 reports, 36 news excerpts, and 36
abstracts. The 628 held files contained 309 rewrites and 319 originals or
ordinary revisions. The panels reuse the previous source-disjoint
evaluation roster, including separate report works, news, abstracts,
eLife excerpts, and dated journal changes. A correct word range covers at
least half the changed words and has word intersection over union of at
least 0.30. The previous paired word-range research check is the comparator.
[Held sources and original comparison](writing-independent-range-transfer-2026-11-15.md)

## Local results

| Check | Correct word locations, tighter → wider | Ordinary files flagged, tighter → wider | Correct word ranges when given the edited sentence |
| --- | ---: | ---: | ---: |
| Previous word-range check | 5 → 6 / 309 | 25 → 86 / 319 | 43 / 309 |
| Absolute predictability | 0 → 1 / 309 | 59 → 96 / 319 | 20 / 309 |
| Difference from both flanks | 1 → 1 / 309 | 38 → 88 / 319 | 21 / 309 |
| Require both scores | 1 → 1 / 309 | 43 → 85 / 319 | 22 / 309 |

All 309 rewrites supplied eligible ranges. At the wider point, the
absolute and combined checks found the same additional report location.
The two-flank check found a different location in an abstract. Each
candidate missed all six useful locations from the previous check. The
union with either of the best candidates would reach 7/309 rewrites and
flag 149/319 ordinary controls, an increase of one correct location and 63
ordinary flags. Under the tighter points, the two-flank union adds one
useful location and 35 ordinary flags. The results support keeping
word-level predictability out of the app's findings. The comparison is
limited to single-sentence rewrites in short, controlled excerpts.
[Score and outcome receipt](evaluation-panels/writing-curvature-ranges-2026-11-15.json.gz)

## Replay

The public receipt includes source hashes and scalar readings at 94,073
word positions. Word coordinates and truth labels come from the earlier
receipts and are checked against the source before native inference.
The receipt omits original prose, tokenizer IDs, and model weights.
The portable reviewer recalculates candidate ranges, ordinary points,
and the union outcomes. Its output agreed byte for byte on Python 3.12
and 3.14.

```sh
export PYTHONPATH=backend
module=scripts.writing_curvature_range_transfer
receipt=docs/evaluation-panels/writing-curvature-ranges-2026-11-15.json.gz
python3 -B -S -m "$module" review "$receipt"
```

Use the [same source-roster arguments](writing-independent-range-transfer-2026-11-15.md#independent-replay)
except `--assets`, and provide the pinned causal snapshot as `--model`.
The original pass used MLX 0.32.4, mlx-lm 0.31.3 and Transformers 5.22.1,
with four-bit affine quantization and groups of 64. If the MLX
environment is separate, make the backend's extraction dependencies
available after the MLX environment's own packages.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$mlx_python" -B -m "$module" capture "$run/curvature.json.gz" \
  --model "$snapshot" --reports "$report_roster" --matched "$matched_roster" \
  --additional "$additional_roster" --dated "$dated_revisions" --elife "$elife_plan" \
  --fresh "$sealed_replication" --gemini "$gemini_shard" --gpt "$gpt_shard"
```

The `audit` command takes the public receipt and an output file, followed
by the same source and model flags. A separate native pass reproduced
the word readings on 16 fixed calibration and held excerpts.
[Native audit](evaluation-panels/writing-curvature-ranges-audit-2026-11-15.json)
