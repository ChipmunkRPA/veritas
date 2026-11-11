# Genuine Turkish speech controls from VoxForge

Use the original publisher's public 48 kHz, 16-bit Turkish submission
directory. The publisher identifies these as volunteer read speech and
provides speaker-session packages with prompts and license notices.
Each selected session must carry an explicit GPL grant for its content
and declare both Turkish and a contributor matching the frozen listing.
[Publisher archive](https://repository.voxforge1.org/downloads/tr/Trunk/Audio/Original/48kHz_16bit/) · [Submission format](https://voxforge.org/home/submitspeech/linux/step-3) · [Per-session license example](https://www.voxforge.org/home/downloads/speech/turkish-speech-files/onderkuscu-20170414-kez)

Freeze the exact captured listing, tool and historical identity commitments
before acquiring media. Use a fixed salted ordering of case-folded named
contributor groups and then of their sessions. Keep twelve groups and at
most three sessions per group. Exclude anonymous entries and known
VoxForge contributor keys. These groups are declared contributor aliases.
They do not establish verified human identities or pretraining disjointness.

Acquire only the frozen publisher HTTPS URLs. Keep the complete bytes and
the HTTP receipt or failed attempt for each session. Verify gzip and tar
byte bounds before inspecting the contents. Inspect the session metadata,
prompt rows and native WAV/FLAC formats. Select the first qualifying
clip in the fixed order within each group. Keep distinct normalized
prompt hashes and exclude exact original and production PCM matches in
the recorded project history and already qualified cases. Unavailable
sessions and clips remain in the attempt journal and unfilled contributor
slots remain in the planned denominator.

Require native mono 48 kHz signed-16-bit audio between 2.5 and 4 seconds.
Decode the entire selected interval through the production decoder,
verify continuous zero-origin 16 kHz mono PCM, and match the expected
native duration to within one output sample. Keep the exact archive,
member, prompt, permission notice, WAV and production PCM commitments.
No clipping, padding, inference or user-facing changes enter preparation.
The source and prepared WAVs stay in the local private run.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_voxforge_tr_genuine_v188 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_voxforge_tr_genuine_v188 acquire
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_voxforge_tr_genuine_v188 prepare
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_voxforge_tr_genuine_v188 readout
```

Acquisition is separate from the offline v108 data-only preparation.
The readout checks the guarded receipt and saved production PCM, and
publishes counts without source identifiers. The panel supplies genuine
Turkish controls for a future language-transfer comparison. It does not
establish Turkish calibration or a population false-alarm rate.
