# Genuine Turkish speech transfer pilot

Bind the exact [corrected v192 VoxForge preparation](audio-voxforge-tr-genuine-v192-method.md),
its successful data-only guard, and the production WAV and PCM
commitments. Three distinct declared contributor groups qualified
across the 12 frozen slots. The other nine slots remain visible as
preparation unavailable. The original v188 attempt and its acquisitions
remain bound through the v192 freeze.

Score each of the three eligible clips once through the same
[v179 English-reference route](audio-alania-transfer-v179-method.md).
Use its variable-length production speech window, frozen classifier,
calibration and lazy original-side stability check. Keep a separate
count for speech-gate rejection. Report all four directions over both
the 12 planned slots and the three qualified slots. An AI tendency on a
genuine clip is a directional error. A source or route unavailable
result stays unavailable.

Display the previously published [Alania result](audio-alania-transfer-v179-result.md)
alongside the new genuine pilot. The two panels have different recording
and production histories. Three qualifying voices and nine unavailable
slots make this a descriptive pilot. It cannot estimate a population
false-alarm rate, validate Turkish calibration, or isolate a language
effect. Keep the English reference and customer defaults unchanged.

Use the metadata-only preflight to verify the frozen receipts and WAV
structure. It loads no model and does not initialize app settings. The
guarded worker verifies that its process is still free of app settings and
model imports before loading the route for its first qualified input.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_voxforge_tr_transfer_v191 preflight
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_voxforge_tr_transfer_v191 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_voxforge_tr_transfer_v191 run
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_voxforge_tr_transfer_v191 readout
```

The worker runs offline under the unchanged v108 neural resource guard.
It accesses no holdout source or media. The readout contains counts and
commitments. Individual waveform receipts and scores stay private.
