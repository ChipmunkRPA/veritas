# Learning to locate revised sentences and words

This local experiment compares two versions of a compact English writing
localizer. Both use the same sentence and word scoring heads. One learns
the heads while holding the encoder fixed. The other updates the encoder
and both heads together. Each version reads one submitted text at a time.
[HACo-Det's sentence and word experiments](https://aclanthology.org/2025.acl-long.1069/)
· [Clause-level provenance in Pangram 4](https://arxiv.org/html/2607.27183v1#S3.SS5)
· [Pinned Apache-2.0 initialization](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4/tree/387825ce42dbb39b87911cdf8e383ee3b25184f8)

## Input and training

The source plan takes 800 paired government reports from two OpenAI
generators. It balances four editing stages within each generator. An
ordinary source must match the official report's non-whitespace text and
pre-date 2021. Where a publisher's sentence labels, tags and character
spans agree, the trial aligns its words with the original sentence.
Destination words inserted or replaced in the alignment are positive
training targets. Words retained inside revised sentences receive no word
target. The sentence target records whether an actual revision covers the
sentence. The fit uses 5,106 overlapping windows with disjoint scored
centers, preserving the native character positions in longer files.
[OpAI-Bench data and revision operations](https://arxiv.org/html/2606.06481v1#S3)
· [Benchmark reuse terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench#licensing-and-reuse)
· [GovReport terms and source edition](https://gov-report-data.github.io/#license)

The 96 calibration reports and the held reports have separate original
documents, exact input hashes and normalized 24-word source sequences.
The held roster contains 96 Gemini reports absent from the earlier
project exposure receipts, 28 retained Qwen reports and nine separate
historical prose and economic sources. Calibration includes 55 reports
used in earlier project studies on other methods. The two held publisher
generators supply no training or calibration examples for this fit.
Encoder pretraining overlap is unknown.

## Frozen comparison

A sentence response can use its pooled representation, the strongest
quarter of its word responses, or the mean of the two. Each reading
gets its own three operating points from the ordinary calibration file
maxima. The separate edited calibration cases choose one preferred
reading and point for each encoder. The utility rewards two per correct
location and deducts four per ordinary file flagged and one per misplaced
highlight. All points and the choice are fixed before held inference.
The receipts retain results at each point, including where the candidate
adds locations to the existing passage rule.

A qualifying location covers at least half of a known edit and reaches
0.30 intersection over union in non-whitespace characters. Each edit
can count once. Reports retain missed locations, misplaced highlights,
ordinary file alerts and eligible coverage. A reading can highlight up
to six original sentences per file. Responses are ranking measurements.

## Held comparison

Each cell gives correctly located edits, followed by the number of
ordinary files flagged. The three cohorts contain 266 files in total.
Every file has usable sentence responses.

| Cases | Current app passage rule | Fixed encoder, selected strict point | Tuned encoder, medium joint point | Tuned encoder, selected wider joint point |
| --- | ---: | ---: | ---: | ---: |
| 96 fresh Gemini report pairs | 0/705 · 8/96 | 2/705 · 6/96 | 20/705 · 6/96 | 76/705 · 34/96 |
| 28 retained Qwen report pairs | 1/188 · 2/28 | 0/188 · 1/28 | 4/188 · 1/28 | 18/188 · 5/28 |
| Nine retained sources outside the government report genre | 10/18 · 0/9 | 0/18 · 6/9 | 0/18 · 8/9 | 0/18 · 9/9 |

The medium joint point finds 20 locations and misses 685 in the fresh
Gemini group. Its eight misplaced highlights on edited reports are
counted separately. Fourteen of the 20 correct locations are in the
compression-stage reports. Using it alongside the current check finds
the same 20 fresh locations and flags 12 ordinary reports. At the wider point the
combined checks find 76 locations and flag 38 ordinary reports. On the
retained Qwen reports the medium combination finds five locations and
flags three ordinary reports. The small non-government transfer group
shows a strong domain effect. The medium point was fixed for the
separately generated report test described below.
[Training and source audit](evaluation-panels/writing-token-supervision-fit-2026-11-17.json)
· [Frozen decisions](evaluation-panels/writing-token-supervision-calibration-2026-11-17.json.gz)
· [All held locations and counts](evaluation-panels/writing-token-supervision-held-2026-11-17.json.gz)

Two standalone Python readers reproduced the held matches after the
model ran. A separate CPU reading of 24 calibration files reproduced
the selected locations and agreed with the accelerator scores within
0.00001 logits. Removing the truth fields from inference inputs yielded
exactly the same calibration values and selection. The independent
export verifies all 266 held input bytes and source bindings, then
provides a manifest, Unicode character labels and predictions for
comparison with another detector.
[Independent input and location receipt](evaluation-panels/writing-token-supervision-independent-2026-11-17.json)

## Further transfer to separately generated reports

After the first comparison, the medium joint point was fixed for a new
test using 64 unused government reports. Each input starts with a
12-sentence excerpt from the verified official archive. Two local
Apache-2.0 editors each received 32 source excerpts. The input specified
two source sentences to paraphrase. The output was admitted when the
edit stayed inside the exact replacement interval, the remaining
sentences stayed intact, and any copies of entire earlier sentences
were excluded from the edit labels. This allowed the two-sentence source
interval to expand into additional sentences. Nine generated responses
failed the geometry, length or usable-change check. The remaining 55
pairs contain 149 labeled edited sentences and 55 matched ordinary files.
[First local editor and terms](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306)
· [Second local editor and terms](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674)

| Separate transfer | Current locations | Medium joint locations | Current → medium ordinary flags |
| --- | ---: | ---: | ---: |
| Qwen offline paraphrases | 4/95 | 0/95 | 0/31 → 1/31 |
| SmolLM2 offline paraphrases | 1/54 | 0/54 | 0/24 → 0/24 |

All 110 files completed inference. The medium joint check placed one
misplaced range in an edited file and supplied zero additional correct
locations to the existing check. A post-test diagnostic at the wider
point found three correct locations with 13 ordinary files flagged and
12 misplaced ranges in edited reports. Taking the highest reading per
file with no operating point placed a useful range in 12 of 55 edited
files. The tuned model stays in evaluation. The gains on the first
held report group failed to carry over to short paraphrases made with
the two local editors.
[Separate transfer inputs and positions](evaluation-panels/writing-token-supervision-transfer-2026-11-17.json.gz)

## Reproduce the local comparison

The new source builder checks all pinned shard digests, the official
archive and the earlier transfer file. Name the five source CSVs
`{split}-reports_{generator}.csv` under `$shards` using the pins in the
builder. Supply the earlier native `older-plan.json.gz` from the
[entropy comparison](writing-entropy-prominence-2026-11-17.md#independent-replay).
The source text, original shards, and resulting checkpoints stay local.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
study=scripts.writing_token_supervision_eval
reference=docs/evaluation-panels/writing-token-supervision-independent-2026-11-17.json
"$runner" -B -m scripts.writing_token_supervision_data "$shards" "$gao" "$earlier" "$out/plan" \
  --reference "$reference"
"$runner" -B -m scripts.writing_token_supervision fit "$out/plan" "$checkpoint" "$out/fit" --device mps
"$runner" -B -m "$study" calibrate "$out/plan/calibration.json.gz" "$checkpoint" "$out/fit" "$out/freeze.json.gz" --device mps
"$runner" -B -m "$study" baseline "$out/plan/held.json.gz" "$incumbent" "$out/baseline.json.gz"
"$runner" -B -m "$study" held "$out/plan/held.json.gz" "$checkpoint" "$out/fit" "$out/held.json.gz" \
  --freeze "$out/freeze.json.gz" --comparator "$out/baseline.json.gz" --device mps
python3 -B -S -m "$study" review "$out/freeze.json.gz" "$out/held.json.gz"
"$runner" -B -m scripts.export_writing_token_supervision \
  "$out/plan/held.json.gz" "$out/fit/fit.json" "$out/freeze.json.gz" "$out/held.json.gz" "$out/independent"
```

The second test uses the same saved fit and calibration. Use the local
MLX runtime and generator assets from the [report generation study](writing-report-confirmation-2026-11-16.md).

```sh
transfer=scripts.writing_token_supervision_transfer
reference=docs/evaluation-panels/writing-token-supervision-transfer-2026-11-17.json.gz
"$runner" -B -m "$transfer" prepare "$gao" "$out/plan" "$out/freeze.json.gz" "$out/transfer-sources" \
  --reference "$reference"
for model in qwen smol2; do
  backend/data/text-upgrade/mlx-env/bin/python -B -m scripts.writing_report_sources generate \
    "$out/transfer-sources/$model.json.gz" "$model" "$out/$model-edits.json.gz"
done
"$runner" -B -m "$transfer" assemble "$out/transfer-sources" "$out/qwen-edits.json.gz" "$out/smol2-edits.json.gz" \
  "$out/transfer-inputs.json.gz"
"$runner" -B -m "$study" baseline "$out/transfer-inputs.json.gz" "$incumbent" "$out/transfer-baseline.json.gz"
"$runner" -B -m "$transfer" run "$out/transfer-inputs.json.gz" "$checkpoint" "$out/fit" \
  "$out/freeze.json.gz" "$out/transfer-baseline.json.gz" "$out/transfer-results.json.gz" --device mps
python3 -B -S -m "$transfer" review "$out/freeze.json.gz" "$out/transfer-results.json.gz"
```
