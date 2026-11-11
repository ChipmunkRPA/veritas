# Fresh named VoxForge Turkish contributors

The earlier VoxForge preparation qualified three of 12 contributor slots.
The serving speech gate declined two of those three. This new preparation
freezes 24 new named contributor slots from the saved VoxForge Turkish
publisher listing. It excludes all 12 prior v188 contributor keys and any
VoxForge key in the v58 source history. At most two publisher sessions per
contributor are selected in a fixed hash order. The plan is frozen before
media acquisition. Available identities cover contributor aliases and prior
recorded artifacts. [Original publisher listing](https://repository.voxforge1.org/downloads/tr/Trunk/Audio/Original/48kHz_16bit/) · [Earlier genuine transfer](audio-voxforge-tr-transfer-v191-result.md)

Reuse the original bounded HTTPS acquisition and exact archive verification.
An admitted archive needs a matching named Turkish contributor, a separate
explicit GPL-3.0-or-later session grant, an audited stock GPL reference, and
an attributable prompt manifest. The project retains the publisher bytes,
receipt, prompt hash and license notice privately. The strict v192 check
rejects conflicting terms and unfamiliar stock references. Try at most 24
WAV or FLAC members per session in fixed hash order. Reject recorded v58
and v192 byte, PCM and applicable prompt matches, plus duplicate accepted
inputs within the new panel. [Publisher submission format](https://voxforge.org/home/submitspeech/linux/step-3) · [Rights method](audio-voxforge-tr-genuine-v192-method.md)

Accept supported mono uncompressed PCM WAV and FLAC with a declared sample
count, 8–192 kHz native sample rate, and three to 20 seconds of audio under
the existing two-megabyte member cap. Decode the complete source using the
production mono signed-16-bit 16 kHz path. Verify the native duration against
the production extent and require a continuous source clock. A recording
whose decoded length is already three to four seconds keeps every sample
if it passes the activity rule. For longer speech, choose a contiguous
three-to-four-second excerpt on a 20 ms grid. Both cut points must have two
quiet frames on each side where available.

The activity proxy marks each 20 ms frame whose RMS meets the greater of
−48 dBFS and 25 dB below the recording’s 90th-percentile frame RMS. At least
60% of complete frames must meet the threshold. For crops, try 3.50 seconds
first, then 0.10-second steps outward through 3.00 and 4.00 seconds, shorter
first at equal distance. At the first viable duration, choose the interval
with the most active frames and use the earliest start for ties. This is
an energy and pause rule. Loud noise can qualify and quiet speech can fail.
The experiment measures performance on short, activity-selected read speech.
Report natural short sources separately from crops and retain all unavailable
slots and rejected attempts. The procedure uses neither a learned speech
gate nor detector scores to qualify inputs.

The private receipt links original bytes, full decoded PCM, the cropped
sample extent on the original timeline, the WAV’s zero-origin clock, and
the exact crop PCM and WAV hashes. It also records the activity threshold
and active-frame share. The public readout includes only aggregate counts,
resource observations and artifact commitments. Acquisition and preparation
are separate stages. Preparation runs offline and accesses no holdout
media or model.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_voxforge_tr_genuine_v199 preflight
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_voxforge_tr_genuine_v199 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_voxforge_tr_genuine_v199 acquire
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_voxforge_tr_genuine_v199 prepare
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_voxforge_tr_genuine_v199 readout
```
