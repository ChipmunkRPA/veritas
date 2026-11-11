# Turkish synthetic speech transfer

Use the 24 distinct designed voices and text documents frozen before
scoring in [Alania v178](audio-alania-roster-v178-method.md). This panel
contains 18 reference renders and six reference-plus-instruction renders.
All voices and text are synthetic. Use clean audio from the exact original
CC BY 4.0 domain-text shard. Retain attribution to PatientDesk AI and the
publisher's terms. [Dataset and license](https://huggingface.co/datasets/cloud0day3/alania-synthetic-speech-tr)

The data-only worker verifies the full shard digest and the selected
metadata at the frozen Parquet row positions. It projects the clean audio
column for the selected row groups and extracts only the selected rows.
Decode each embedded FLAC in full through the production audio path. Require
a native 48 kHz stream and 2.5–4 seconds of continuous, zero-origin mono
16 kHz production PCM. Confirm the full native duration against the output
sample count to one output sample. Allow the declared metadata duration to
differ by up to 10 ms plus one output sample for display rounding. Cases
that cannot be decoded on this clock stay unavailable in the denominator.
Only the prepared WAVs and private commitments are stored in the run.

The offline scoring worker verifies the preparation receipt and WAV/PCM
commitments. It uses the pinned full-profile speech gate, candidate voice
classifier and English reference calibration from
[v174](audio-matched-edit-diagnostic-v174-method.md). An accepted waveform
contributes one complete speech window with its native analysed length.
Only the existing original-side stability check can add a model forward.
No fitting, threshold changes or repetition cues enter the study.

Report scored coverage, all four directions, and missed AI cases for all
planned positives and separately for each render mode. A miss includes
any planned case without an AI tendency, including VAD rejection,
inconclusive, or preparation unavailable. An original tendency on these
known positives is a directional error. This is a positive-only,
out-of-language transfer test of one generator on clean speech with the
publisher's high-frequency post-filter. It provides no Turkish calibration,
false-alarm estimate, noise robustness, or generalization estimate across
voice generators.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_alania_transfer_v179 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_alania_transfer_v179 prepare
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_alania_transfer_v179 prepare-readout
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_alania_transfer_v179 run
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_alania_transfer_v179 readout
```

Both workers use the existing v108 guard in separate stages. The data stage
makes zero model calls. The final readout validates resource and scoring
receipts and contains aggregate results without row or voice identities.
