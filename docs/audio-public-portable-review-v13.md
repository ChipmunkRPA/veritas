# Reproduce the rejected audio result without model downloads

The [v13 audio comparison](audio-wavlm-v13-development-evaluation.md) can now be
checked in a separate numerical-only Python environment. The command reads the
already published feature packets, fitted objects and source commitments. It
requires no app setup, pretrained weights, source recordings, transcripts or
private manifests. Its import policy rejects the app, PyTorch, Transformers,
audio/image decoders and plotting packages.

This is an entry point for outside reviewers. It does not claim that an outside
reviewer has participated: the recorded count remains zero.

## Run from the repository root

Use Python 3.12 and a new output path. Create the environment separately from
the app. Installing the seven pinned numerical packages requires network access;
the actual replay uses only files already in the repository.

```sh
python3.12 -m venv /tmp/chipmunk-audio-review
/tmp/chipmunk-audio-review/bin/python -m pip install --isolated \
  -r backend/requirements-audio-public-review.txt
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /tmp/chipmunk-audio-review/bin/python \
  backend/scripts/review_audio_public_portable_v13.py \
  docs/evaluation-panels/audio-v13-development-review.json.gz \
  /tmp/new-portable-audio-review.json --refit
```

The feature packets and references must stay beside the main packet. An
alternative reference directory can be passed with `--references`. No fitting,
calibration or development input may be omitted. The program refuses to
overwrite an existing review and exits unsuccessfully if a check fails. Omitting
`--refit` still checks outcomes and normalization/calibration but does not refit
the classifier; the receipt explicitly records that difference.

## What is reconstructed

- All 4,248 feature records in their original order, reproducing the complete
  413,255,997-byte original receipt by SHA-256. The transport preserves float32
  values, signed zero and the original JSON representation.
- Fitting-only weighted normalization, each classifier's own group/truth
  calibration, all 1,449 development outcomes, both stability checks and the
  required 511 filtered probes' commitments.
- Complete condition/generator statistics, uncertainty calculations and all
  sixteen unchanged adoption requirements for both candidate arms.
- With `--refit`, both complete fitted objects using the original fixed learner
  on fitting sources only. Only four checksum-verified numerical definitions
  are extracted from that source; its neural imports and command entry point
  are never executed.

The new probability/normalization/calibration reconstruction is separate from
the inference path. Statistical summaries, bootstrap/gates, guard helpers and
the declared refit learner remain shared project implementations. This is not
an independently designed statistical method. Earlier experiment packets are
checksum-bound inputs; this lighter command does not numerically replay every
earlier experiment. The 593 native audit cases are checked as recorded metadata,
not rerun through neural models. Source re-execution and fresh predictive
evaluation are separate work.

## Local verification and limits

The [recorded replay](evaluation-panels/audio-v13-portable-review.json) passed in
a newly created environment without system site packages, PyTorch, Transformers
or app dependencies. Model/data cache locations were empty. Both complete
fitting-only objects matched exactly and every outcome/statistic reproduced.
The [environment receipt](evaluation-panels/audio-v13-portable-environment.json)
records actual installed wheel hashes and versions. Fourteen isolated reviewer
checks plus thirteen transport tests passed. They include hand-computed margins,
blocked imports, malformed features, unassessed inputs, stability vetoes and
changed source/transfer/object commitments. Transfer tests do not establish
predictive accuracy.

The tested runtime is Python 3.12.6 on macOS arm64 with NumPy 2.2.6, SciPy
1.18.1 and scikit-learn 1.9.0. Exact optimizer objects may differ with another
numerical library, BLAS implementation or architecture. Report such differences
with the commit, runtime, failure and receipt; do not silently relax comparison
tolerances or change the original fitted objects. Cross-platform exact refits
have not been demonstrated.

The reproduced result still rejects both candidates. No arm is selected for
fresh confirmation, and the current service is unchanged. All 1,080 reserved
confirmation inputs remain unscored. Derived speech features retain their
attached source terms and may contain voice information; absence of raw audio
does not establish anonymity. This work adds reproducibility, not new detection
accuracy, independent human review or court evidence.
