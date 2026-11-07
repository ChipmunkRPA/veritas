# Portable public mouth-evaluation review

The complete rejected V12 result can now be reconstructed and refitted without
the app, neural libraries, original training modules, video decoders, recordings
or pretrained weights. This makes public review easier; it does not change the
result:26/32 edits located with12/56 ordinary clips flagged, versus14/32 and0/56
for the retained reference. No model is admitted and no unused actor is scored.

## Two review paths

The default path uses Python's standard library only, with site packages disabled:

```sh
PYTHONPATH=backend python -S -m scripts.review_mouth_temporal_v12 \
  docs/evaluation-panels/mouth-v11-fitting-review.json.gz \
  docs/evaluation-panels/mouth-v12-execution-freeze.json \
  docs/evaluation-panels/mouth-v12-development-review.json.gz /tmp/review.json
```

It checks the original public source checksum and arithmetic, all100 stored
trees,492 ordinary point windows,2,321 ranked development windows,88 outcomes,
source-time overlaps, evidence priority, counts and all fixed requirements.
It does not recalculate source-mask intersections or execute neural forwards.

For the optional complete fitting-only JSON tree refit, use Python3.12.6 in an
isolated environment with the same seven numerical versions:

```sh
python -m venv /tmp/mouth-review-env
/tmp/mouth-review-env/bin/python -m pip install --isolated --no-cache-dir \
  -r backend/requirements-mouth-public-review.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=backend \
  /tmp/mouth-review-env/bin/python -m scripts.review_mouth_temporal_v12 \
  docs/evaluation-panels/mouth-v11-fitting-review.json.gz \
  docs/evaluation-panels/mouth-v12-execution-freeze.json \
  docs/evaluation-panels/mouth-v12-development-review.json.gz /tmp/refit.json --refit
```

The new numerical helper separately rebuilds actor/class/condition weights,
the full3,560×49 fitting matrix, declared estimator and complete JSON tree
export. It uses the same scikit-learn learner; it is not an independently
implemented boosting algorithm. Exact equality against the complete stored
fitted object remains required. The old original-training-module import has
been removed from this review path. No probability point, model parameter,
source role, recorded outcome or advancement requirement changes.

## Local verification and external-host plan

Both paths passed locally. The fitting-only refit reproduced every stored tree
exactly in the separate numerical environment, with app/neural/decoder imports
and all three original training modules explicitly blocked. That environment
has no system site packages and none of the forbidden packages installed.
The [environment receipt](evaluation-panels/mouth-v12-portable-environment.json),
[exact-refit receipt](evaluation-panels/mouth-v12-portable-refit-review.json) and
[standard-library receipt](evaluation-panels/mouth-v12-portable-stdlib-review.json)
preserve the checks. Seven malformed fitting-input checks and six existing
scalar-review checks also passed. The earlier native/refit receipts remain
unchanged and retain their historical reviewer hashes.

The new `mouth-public-replay.yml` workflow runs separate standard-library and
numerical-refit jobs on GitHub-hosted Ubuntu24.04/Python3.12.6. It has read-only
repository permissions, disables persisted checkout credentials, pins the two
actions by commit and downloads no neural models or source recordings. Its
results are pending until the actual hosted jobs complete. A failure will remain
visible; local success does not imply exact cross-platform refitting.

This is reproducibility preparation and automated external-host testing, not
outside human participation or independent source-truth/accuracy validation.
No outside human reviewer has participated. The rejected detection result,
raw-media/privacy boundaries and separate fresh-confirmation requirements
remain intact.
