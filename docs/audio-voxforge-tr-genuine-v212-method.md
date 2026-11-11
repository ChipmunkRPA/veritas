# Additional Turkish speech groups for a shared-background comparison

Select 16 named VoxForge Turkish contributor groups prospectively from the
same saved publisher listing. Exclude all 12 v188 and all 24 v199 planned
groups, including those that failed preparation, along with known VoxForge
keys in the v58 history. Fix the session order and at most two sessions per
group before download. Never replace a failed contributor with another
group. Contributor aliases establish known group separation within the
available project records. [Publisher listing](https://repository.voxforge1.org/downloads/tr/Trunk/Audio/Original/48kHz_16bit/) · [Prior intake](audio-voxforge-tr-genuine-v199-result.md)

Keep the v199 acquisition bounds, strict v192 explicit per-session GPL
permission checks, WAV and FLAC qualification, and non-neural three-to-four-
second activity and pause rule. Original bytes, full production PCM, crop
PCM and original-to-playback sample mapping stay bound to each private
receipt. Reject exact prior prompt hashes, original byte hashes and either
full or crop PCM hashes for all 22 qualifying v192 and v199 sources. Apply
the existing history and within-panel checks. The earlier v192 prepared
waveform covers its complete original source, so its PCM commitment serves
as both the full and short identity. Earlier preparation metadata supplies
these exclusions without reopening earlier audio. [Rights method](audio-voxforge-tr-genuine-v192-method.md) · [Crop method](audio-voxforge-tr-genuine-v199-method.md)

The guarded preparation records every planned slot, separates full short
files from pause-bounded excerpts, and verifies saved WAVs during readout.
Publish aggregate counts and commitments. Individual prompts, aliases,
recording identifiers, media, candidate lists and source offsets stay
private. Preparation yields activity-selected Turkish read speech for a
separate matched background experiment. It makes zero model calls and
does not open or score holdout media.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_voxforge_tr_genuine_v212 preflight
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_voxforge_tr_genuine_v212 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_voxforge_tr_genuine_v212 acquire
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_voxforge_tr_genuine_v212 prepare
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_voxforge_tr_genuine_v212 readout
```
