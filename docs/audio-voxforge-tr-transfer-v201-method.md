# Fresh genuine Turkish speech transfer

Bind the [v199 genuine VoxForge preparation](audio-voxforge-tr-genuine-v199-method.md)
and its exact guarded result. All 34 selected session archives were acquired.
Keep the 24 frozen contributor slots. Eighteen distinct groups qualified
with pause-bounded excerpts and one qualified with an entire natural
short clip. Retain the five unavailable source slots. All qualified
recordings carry the declared GPL-3.0-or-later session grant.

The model-free preflight verifies the v199 freeze, private case and
acquisition commitments, successful preparation guard, original sample
clocks and exact 19 prepared WAVs. It verifies the two earlier comparisons
from their guarded receipts without reopening their audio. The scoring
worker accesses only the 19 prepared WAVs and runs offline under the
unchanged v108 neural resource guard.

Run the same English-reference route used in v191 and v179. Apply its
production speech gate to each complete three-to-four-second prepared
clip. Use the unchanged classifier, calibration and original-side
stability check. Report planned, eligible, speech-gate-rejected and
scored counts overall and by selection class. Give AI, original,
inconclusive and unavailable counts across planned and eligible slots.
Keep the five slots without an eligible selection in their own class.
Count AI direction on an admitted genuine source as a directional error.

Show the earlier v191 genuine pilot and v179 synthetic-only results
alongside this panel. The earlier pilot scored one clip after two speech
gate rejections among its three eligible inputs. The synthetic panel
contains 24 clips from a different source and production process.
These panels differ in sampling, recording conditions and selection.
Within v201, the natural-short class contains one clip. The comparisons
are descriptive and cannot establish Turkish calibration, the causal
effect of cropping, or population false-alarm and miss rates.
[Earlier genuine result](audio-voxforge-tr-transfer-v191-result.md) · [Synthetic result](audio-alania-transfer-v179-result.md)

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_voxforge_tr_transfer_v201 preflight
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_voxforge_tr_transfer_v201 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_voxforge_tr_transfer_v201 run
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_voxforge_tr_transfer_v201 readout
```

The readout verifies the private outcome journal and guard before
returning aggregate counts and commitments. Individual source clocks,
scores and recordings remain private. The study makes no threshold,
calibration, or customer setting changes and accesses no holdout.
