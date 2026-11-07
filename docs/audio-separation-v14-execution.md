# Two-source audio: frozen execution

6 November 2026. The complete local experiment now binds 79 project files,
six installed implementations, all 4,248 original consumed inputs, exact voice/VAD
and separation weights, source roles, own fitting/calibration, stability and the
original sixteen requirements. At freeze publication no project recording had
been scored or used to fit this candidate. The complete
[protocol](audio-separation-v14-protocol.md) and
[public freeze](evaluation-panels/audio-v14-separation-freeze.json) were published
and remote-verified as `220e4888942e00ac934150945ce1fccbab520e29` before
extraction started. Extraction is now running under the unchanged lock; fitting
and calibration have not started.

Private lock SHA-256:
`93223afb27132aa3557b70f5c433aec126409ebd06d2beea6d5f34b3551e744b`.
All original waveform files were rehashed before the lock was written. The
1,926 fitting / 873 calibration / 1,449 development roles and 62 unassessed
inputs are unchanged. All 1,080 reserved confirmation variants stay unscored.

The 67 selected tests pass: 42 separation/learner/execution tests and 25 prior
backbone checks. Tests include the complete unassessed comparison path, required
probe refusal, exact estimate clocks and head/feature binding, malformed fitted
mappings, source permutations and numerical gradients. Thirteen alterations to
the actual completed freeze are rejected before recording inference.
[Alteration receipt](evaluation-panels/audio-v14-freeze-alterations.json).
An initial preflight found a missing output-directory setup step. It was fixed
before the completed lock or any project recording forward; the failed local
attempt is retained. These checks establish preparation, not detection accuracy
or outside human review.

## Run locally

Set the existing local model/data/cache directories, commercial licence policy,
offline loading and four CPU threads. Use the exact original source artifacts,
pinned separation assets and verified upstream wheels documented in
[preparation](audio-separation-v14-preparation.md). No dependency installation or
remote model code is performed by the runner.

```sh
PYTHONPATH=backend backend/.venv/bin/python -m scripts.run_audio_separation_v14 freeze \
  /path/to/audio-v9 /path/to/new-audio-v14/lock.json \
  --weights-root /path/to/separation-v14-convtasnet \
  --package-root /path/to/verified-upstream-wheels
PYTHONPATH=backend backend/.venv/bin/python -m scripts.run_audio_separation_v14 public \
  /path/to/audio-v14/lock.json /path/to/new-public-freeze.json
```

Publish and verify the complete prospective freeze first. Then run the following
stages separately, retaining every journal. The existing voice loader is the sole
owner of inter-op initialization; do not add a second caller. Raw source estimates
are hashed and released from memory rather than written as another audio corpus.

```sh
PYTHONPATH=backend backend/.venv/bin/python -m scripts.run_audio_separation_v14 extract \
  /path/to/audio-v14/lock.json /path/to/audio-v14/features.jsonl
PYTHONPATH=backend backend/.venv/bin/python -m scripts.run_audio_separation_v14 train \
  /path/to/audio-v14/lock.json /path/to/audio-v14/trained \
  --receipt /path/to/audio-v14/features.jsonl
PYTHONPATH=backend backend/.venv/bin/python -m scripts.run_audio_separation_v14 compare \
  /path/to/audio-v14/lock.json /path/to/audio-v14/responses.jsonl \
  --receipt /path/to/audio-v14/features.jsonl --head /path/to/audio-v14/trained/head.json
PYTHONPATH=backend backend/.venv/bin/python -m scripts.run_audio_separation_v14 review \
  /path/to/audio-v14/lock.json /path/to/audio-v14/summary.json \
  --receipt /path/to/audio-v14/features.jsonl --head /path/to/audio-v14/trained/head.json \
  --responses /path/to/audio-v14/responses.jsonl
PYTHONPATH=backend backend/.venv/bin/python -m scripts.run_audio_separation_v14 native \
  /path/to/audio-v14/lock.json /path/to/audio-v14/native-review.json \
  --receipt /path/to/audio-v14/features.jsonl --head /path/to/audio-v14/trained/head.json \
  --responses /path/to/audio-v14/responses.jsonl
```

The native stage replays the original 99 cells and every computed filtered probe,
including original voice/VAD behavior, estimate hashes, both raw/features and
independent upstream graph waveform hashes. Separate numerical reconstruction
and exact fitting-only refits remain necessary alongside native correspondence.
A complete result must retain misses, false calls, inconclusive outcomes and
every frozen requirement, including strong background speech. A failed arm
cannot be rescued by changing a point, dropping difficult cases or relaxing gates.

The separate arithmetic reviewer is implemented outside the frozen inference
files. Seven focused tests pass, including hand-computed source/bag responses,
permutation, standardization/intercept positions and stable extreme logits.
Complete real-data reconstruction and exact refits await the completed receipt,
trained objects and outcomes; the tests do not claim those stages have passed.

```sh
PYTHONPATH=backend backend/.venv/bin/python -m scripts.review_audio_separation_v14 \
  /path/to/audio-v14/lock.json /path/to/audio-v14/features.jsonl \
  /path/to/audio-v14/trained/head.json /path/to/audio-v14/responses.jsonl \
  /path/to/audio-v14/summary.json /path/to/audio-v14/numerical-review.json --refit
```

This reconstructs source and bag probabilities, fitting-only normalization,
group/truth calibration, stability and operational counts with separate arithmetic.
Declared statistics/gates and the exact refit learner remain shared. It runs no
neural forwards and is not an outside reviewer.

Only a passing consumed result can permit a separately frozen fresh comparison;
it does not enable the detector, customer confidence or model-specific claims.
The source/model terms and sample-rate metadata limitations remain explicit.
Image, video, writing, outside testing and eventual GCP validation remain in the
wider program. Actual outside human reviewers remain zero.
