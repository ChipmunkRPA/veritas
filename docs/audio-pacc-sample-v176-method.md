# Independent background-noise control preparation

PACC-P v1.0 provides externally authored, genuine speech with matched
acoustic treatments. The publisher releases it under CC BY 4.0 and provides
third-party attributions for VCTK, AMI, MUSAN and FSD50K. The local archive
uses the exact 154,777,600-byte sample and the publisher revision
`b6c9e5f2ff0e3d60f90ade6d37d79245207a4f64`. The publisher tar index and
SHA-256 list agree with the downloaded bytes. The source code checks the
release-level and in-archive credit records before preparation.
[Publisher datacard and terms](https://moonscapesoftware.ca/datasets/pacc-p/) · [Pinned publisher files](https://huggingface.co/datasets/moonscape-software/PACC-P/tree/b6c9e5f2ff0e3d60f90ade6d37d79245207a4f64)

Qualify all 14 microphone inputs at the first three seconds, each clean and
with babble, environmental noise and instrumental music at the publisher's
0 and +5 dB conditions. All six treatments preserve the recorded native
sample count and rate. The publisher measures SNR over the full clip using
active speech level. The three-second prefix may have a different SNR.
The sample has four AMI meeting excerpts and five VCTK utterances recorded
on two microphones. Each matched microphone pair shares its utterance and
speaker. The historical catalogue already names the five VCTK speakers.
Keep them as transfer controls and report AMI separately. Historical
metadata provides limited evidence of independence from prior sources.

Decode these 98 inputs with the production decoder under the data-only
guard. Require exactly 48,000 signed-16-bit mono samples at 16 kHz, a
zero-origin continuous source mapping, and byte agreement between the
decoder output and saved WAV. Keep every planned case, including unavailable
decodes. WAVs, source IDs, sample timings and per-file hashes stay private.
This stage performs zero model calls and measures preparation coverage.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_pacc_sample_v176 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_pacc_sample_v176 run
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_pacc_sample_v176 readout
```

The sample supports a small false-alarm and coverage comparison across
acoustic conditions. Synthetic misses, partial-edit localization and
population calibration require additional source groups.
