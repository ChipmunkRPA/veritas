# Calibrating contiguous word ranges for short writing edits

The previous word-level check marked the top-scoring 60% of one sentence.
This trial asks whether shorter, contiguous word ranges make the selection
more precise and whether scores relative to nearby sentences help distinguish
a rewrite from ordinary changes. Both candidates use the same frozen word
responses from a single submitted excerpt. The experiment uses the retained
word head with the existing Apache-2.0 encoder and no additional weights.
The recent word-level editing study motivates measuring the changed region
separately from overall text classification.
[Prior word-level check](writing-word-localization-2026-11-14.md) ·
[MixDetect research](https://arxiv.org/abs/2609.32625)

## Measurement

Measure contiguous windows of 3, 5, 8, 12, 18 and 24 words in each of the
three interior sentences. The absolute variant scores their mean word
response. The relative variant subtracts the highest same-length mean from
either adjacent sentence. A shorter adjacent sentence contributes its
full mean. A fixed 144-work set of ordinary fit reports supplies the
median and median absolute deviation of each size's file maximum.
The normalized maximum selects one native word range. Ties withhold the
selection. The ordinary report, news and abstract calibration files
independently set their existing 10% and 25% comparison levels. Both levels
are fixed before scoring each held panel.

A useful location covers at least half of the inserted or replaced words
and reaches at least 0.30 word intersection over union. Each generated
excerpt contributes its exact changed-word set, recovered from the source
pair. The inference sees the submitted excerpt alone. The comparison also
counts which useful locations appeared on the same sentence in the matched
ordinary version, and tracks gained and lost cases separately.

The held material uses the same previously tested report, dated journal,
news and abstract works. The separate news and abstract cohorts use the
original evaluation's work and exact 12-word overlap exclusions. Their
generators also appear in the old fit. The GPT-5.4 report generator and
the local Qwen report generator were held out of the original word fit.
[Existing source qualification](writing-word-localization-2026-11-14.md)

## Held comparison

Each cell shows useful AI-rewrite locations and flagged matched ordinary
files at the wider fixed level. For the journal panel, the ordinary files
are the later dated revisions.

| Held panel | Prior word readout | Absolute contiguous ranges | Relative to adjacent sentences |
| --- | --- | --- | --- |
| Report rewrites and ordinary versions | 4/93 · 15/93 | 5/93 · 11/93 | 6/93 · 16/93 |
| Journal rewrites and later editorial revisions | 2/32 · 3/21 | 3/32 · 3/21 | 2/32 · 4/21 |
| News rewrites and ordinary versions | 2/56 · 13/56 | 2/56 · 15/56 | 2/56 · 29/56 |
| Abstract rewrites and ordinary versions | 3/60 · 14/60 | 1/60 · 5/60 | 2/60 · 17/60 |

All useful report locations came from the local Qwen family. The absolute
rule added one useful report location and retained the prior four. Its
flagged report controls dropped from 15 to 11, with four newly flagged
and eight no longer flagged. The relative rule added four useful report
locations and lost two. Two of its six useful report results also selected
the same sentence on the matched ordinary version. In journals, the
absolute rule added two useful SmolLM2 locations, lost one Qwen location
and newly highlighted one dated revision. The absolute rule added no
news or abstract locations. Under the tighter level neither candidate
located a report edit from GPT-5.4.

## Where the GPT-5.4 check fails

An evaluation-only diagnostic supplies the known target sentence after
inference. It asks whether the model's strongest 60% of words inside that
sentence are useful, independently of the file cutoff and whether the
model picked the sentence itself.

Of 60 GPT-5.4 report edits, 57 had complete tokenizer coverage. The model
ranked the edited sentence first in 22 and reached useful word overlap
in five when given the sentence. Only one satisfied both conditions.
Both contiguous variants and the prior word readout found zero useful
locations at the two ordinary-calibrated levels. On the 32 local Qwen
edits, the target-only diagnostic found all 32, while the sentence
ranked first in 16. These two generator families expose different
limitations. A better sentence selector could recover some Qwen edits.
GPT-5.4 needs stronger word-level discrimination even inside the edited
sentence. The contiguous range check remains experimental.

## Independent replay

The public receipt includes numeric range positions, maxima, source hashes,
normalization and operating levels. The earlier public word receipts
provide the exact changed-word labels and prior selections. Source text,
token states and weights stay local. Standard-library replay verifies the
geometry, recalculates both operating levels and recounts gains, losses
and the target-only diagnostic. A second CPU run reproduced the full
receipt byte for byte. The reviewer rejected a shifted word range and
an unknown source group.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
receipt=docs/evaluation-panels/writing-local-word-spans-2026-11-15.json.gz
python3 -B -S -m scripts.writing_local_word_spans review "$receipt"
```

Use the [prior native roster and fitted word head](writing-word-localization-2026-11-14.md#independent-replay)
to repeat all measurements locally.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
backend/.venv/bin/python -B -m scripts.writing_local_word_spans run \
  "$roster" "$matched" "$additional" "$encoder" "$fitted/trained.safetensors" "$run/spans"
gzip -dc "$receipt" | cmp - "$run/spans/receipt.json"
```
