# AMI stereo capture intake

Qualify four original AMI files from the University of Edinburgh release.
Take `Array1-01` and `Array1-05` from `ES2004a` and `IS1009a`. The microphones
belong to the same tabletop array within each meeting. AMI reports that the
recording devices share a timeline and that these WAVs were delivered at
16 kHz after conversion from the 48 kHz capture. The two series are absent
from the project's four previously scored PACC-P AMI meeting contexts.
[ES2004a files](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2004a/audio/)
· [IS1009a files](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/IS1009a/audio/)
· [Capture and delivery notes](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml)

The first-party release licenses the recordings CC BY 4.0. AMI documents
participant opt-in, signed consent and review for research distribution.
The public consent form describes this scope and the possibility of
recognition. Individual signed releases and permissions to use voices as
identity or synthesis references are outside this evidence. Use the original
human speech only for a local genuine-origin evaluation. Keep source audio,
speaker identities and private hashes local. Credit the AMI project and
identify the two-channel extraction in any aggregate description. The old
OpenSLR mirror advertises different terms, so this intake binds the current
publisher's own release. [License](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)
· [Ethics and consent](https://groups.inf.ed.ac.uk/ami/corpus/ethicsandconsent.shtml)
· [Participant consent form](https://groups.inf.ed.ac.uk/ami/corpus/ami_consent_form_081004.pdf)

Freeze the code, FFmpeg binaries and exact primary documentation snapshots
before retrieving audio. The acquisition allows only the four declared HTTPS
URLs and stores their received bytes in the approved Seagate archive. Each
file is limited to 40 MiB. Requests have a 20-second socket timeout and a
180-second per-file deadline. No redirect or automatic retry is allowed.
Record the received byte count and SHA-256 and any HTTP validator supplied.
The first-party directories list the files without publisher checksums, so
the acquisition commitments bind received bytes for local replay.

Require distinct signed-16-bit mono microphone payloads with equal 16 kHz
sample counts and complete source extent. Preserve the original order and
identical sample indices across the two channels. Read 12 consecutive-sample
three-second windows per session at `60 + 30 × ordinal` seconds. Keep quiet
and overlapping speech within that fixed grid. Interleave the actual channel
bytes without gain changes, shifts, padding or source selection. Bind each
output to its half-open original meeting interval and its zero-origin file
timeline. The exact PCM channel and source clock checks validate correspondence
with AMI's published common timeline. They do not independently measure
historical clock drift or remove physical differences in sound arrival time.

Reopen the prepared stereo WAV, verify exact channels, then verify its real
production mono PCM against the explicit FFmpeg resampler. The separate
readout reconstructs the native pairs and source intervals from the original
files. The pilot contains 24 genuine-origin windows across two meeting
sessions. Detector coverage and false flags require a separate experiment.
The intake measures neither AI-speech detection nor population accuracy.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_ami_native_stereo_v291 plan
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_ami_native_stereo_v291 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_ami_native_stereo_v291 acquire
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_ami_native_stereo_v291 qualify
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_ami_native_stereo_v291 readout
```

The plan reads no media and makes no network request. The operator runs the
freeze and acquisition with access to the first-party host and the approved
archive. Qualification and readout use local bytes. Partial attempts remain
in place for inspection. These commands open no model.
