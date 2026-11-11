# Independent replay of the PACC-P genuine-speech controls

This is a new replay path for the 98 genuine inputs and 84 matched comparisons
reported in [v177](audio-pacc-false-alarm-v177-result.md). It leaves the original
[v176](audio-pacc-sample-v176-method.md) and
[v177](audio-pacc-false-alarm-v177-method.md) freezes, scripts and receipts
unchanged. It does not use the project's private historical voice catalogue.
The original report's statement that the five VCTK speakers were previously
catalogued is a historical disclosure, not something this replay independently
verifies. This panel contains genuine controls only and cannot measure misses on
synthetic speech.

The researcher supplies an empty output directory, a local directory holding
the publisher sample and metadata, and a separate model directory. The script
checks the frozen public code and receipt hashes and all publisher inputs before
decoding. It uses every publisher input and condition; it does not select cases
by model outcome. Individual source metadata, prepared audio, raw scores, and
per-file commitments stay in the private output directory. Review privacy and
licensing before sharing these files; the aggregate `comparison.json` contains
condition and transition counts, not individual responses.

## Inputs and rights

Obtain the [PACC-P publisher files](https://huggingface.co/datasets/moonscape-software/PACC-P/tree/b6c9e5f2ff0e3d60f90ade6d37d79245207a4f64)
at revision `b6c9e5f2ff0e3d60f90ade6d37d79245207a4f64`. Put
`pacc-p_sample.tar`, `README.md`, `SHA256SUMS`, `TARBALL_INDEX.csv`,
`attribution.csv`, and `condition_index.csv` in one directory. The archive is
154,777,600 bytes, SHA-256
`878caa7fbc67c06380f99c2e165fd8d306c7c0229ce00e03c475e35076c14b99`.
The [publisher datacard](https://moonscapesoftware.ca/datasets/pacc-p/)
states CC BY 4.0 and provides attribution for source material including VCTK,
AMI, MUSAN and FSD50K. The script checks the frozen attribution inventory;
researchers should follow the publisher's terms and preserve required credits.

Supply these exact local assets without enabling remote model code:

| Local filename | Public source and recorded grant | SHA-256 |
| --- | --- | --- |
| `pellav2_detector.pt` | [Pella v2 checkpoint](https://huggingface.co/Sadanie/pellav2-audio-deepfake-detector/tree/29daea8b04acf2d3f11d1d0da6b01c17fc8a1e4b), publisher MIT declaration | `d21aedfb52e2f87706c2753715a29665482b0f4a034800c31579df5d458559ac` |
| `w2v-bert-config.json` | [W2v-BERT 2.0 configuration](https://huggingface.co/facebook/w2v-bert-2.0/tree/da985ba0987f70aaeb84a80f2851cfac8c697a7b), publisher MIT declaration | `f5572bd5998b68182e9c328a43127ed21fed687f6910497136b91a4e3b0e3675` |
| `silero_vad.jit` | [Silero VAD](https://github.com/snakers4/silero-vad/tree/066409f3d9db45a30d21dfcbf9147826ffdd7052), MIT code and pretrained models | `e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720` |

The Pella checkpoint contains its encoder weights. The declarations do not
document its training recordings or establish model accuracy. See the
[audio notices](../frontend/public/licenses/audio-model-notices.txt) and
[commercial model review](commercial-model-audit.md). The source, frozen
calibration and sample/model hashes are public; the model files are acquired
from their publishers, not bundled by this replay.

## Run locally

Use the project Python environment, ffmpeg and ffprobe. The frozen reference
records package versions and exact original ffmpeg executable hashes. This
replay records its own platform, package versions and executable hashes;
differences mean it is a separate numerical replay, not verification of the
original machine's exact binary execution. Scoring is CPU float32 and loads
the pinned local weights strictly. It can require several GiB of memory; allow
adequate resources and run it independently of other model jobs.

Create an empty directory accessible only to the researcher. Qualification
checks all publisher inputs and prepares the first three seconds of all 98
cases using the production decoder; it makes no neural calls. A failed decode
stays in the planned denominator as unavailable. The scorer requires a
completed qualification and verifies the prepared WAV and PCM hashes again.
It uses the recorded full-profile gate, voice score, calibration and stability
check, and retains gate rejections and scoring errors as unavailable.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.reproduce_pacc_v177 qualify \
  --publisher /path/to/publisher-files --output /path/to/empty-private-output
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.reproduce_pacc_v177 run \
  --models /path/to/pinned-model-files --output /path/to/empty-private-output
```

Each command refuses to overwrite its existing output. `qualification.json`
records coverage, source and local environment commitments; the accompanying
private files bind the deterministic publisher roster and per-case preparation.
`comparison.json` compares only aggregate direction and route-status counts and
clean-to-noisy transition counts against the published v177 receipt, separated
by AMI, VCTK and microphone. It reports mismatches rather than adjusting inputs
or thresholds. Matching aggregate counts are not proof that the original private
per-input results or historical source-independence claims were reproduced.
Nominal noise levels describe the full publisher clip, not necessarily the
three-second prefix. The small set, dependent VCTK microphones and unknown
checkpoint-training overlap limit generalization.

Model-free checks of the replay's input rejection, coverage accounting, output
privacy and aggregate comparison can be run without the dataset or weights:

```sh
PYTHONPATH=backend backend/.venv/bin/pytest -q -o addopts='' backend/tests/test_reproduce_pacc_v177.py
```

## Verified local replay

A separate clean checkout used the documented commands with explicit
publisher, model and output paths. All 98 inputs qualified and all 52
published condition or transition tables matched. The replay used the same
package versions and decoder binaries as the original. A separate local
audit compared private records and confirmed all 98 source hashes, prepared
waveforms, decoded PCM, raw scores and decisions. The replay command itself
does not require those private records. This checks portability of the
entry point on the same Mac. An outside researcher or another operating
system can use the public path to verify their own run.
[Local replay receipt](evaluation-panels/audio-pacc-portable-replay-verification.json)
