# Video speech dropouts under continuing room sound

The existing near-silence check can identify a sudden loss of sound while a
mouth keeps moving. This trial measured whether a stereo recording also
supports locating a loss of speech while room sound continues. The candidate
remains an evaluation tool. Its selected setting found a few edits with
quieter ambience and repeatedly flagged ordinary stereo mixes. The isolated
app prototype also yielded no finding in its first replay.

## Held comparison

The study staged hard mutes and mutes with 25 ms fades. It left recorded
stereo ambience in place and checked the same picture presentation and decoded
sound range in PCM and 48 kHz AAC. Ordinary comparison clips include unchanged
mixes, whole-sound gain, local gain reduction, changes in voice panning and
reduced ambience. The unit of independence is the speaker. Counts for the two
audio encodings represent the same 14 held speakers and edit locations.

| Foreground relative to ambience | Correct locations | False ranges on edited clips | Ordinary controls flagged |
| --- | ---: | ---: | ---: |
| 8 dB above, each encoding | 5/28 | 2/28 | 10/70 |
| 5 dB below, each encoding | 0/28 | 0/28 | 0/70 |

The existing near-silence reading located zero of the staged edits and flagged
none of the ordinary controls. This comparator used a mono mix of the exact
stereo recordings. The new scan found eligible stereo windows in 74 of 98 PCM
and 72 of 98 AAC deliveries with quieter ambience, and 97 of 98 in each
encoding with louder ambience. The five correct locations came from three
speakers. The ten ordinary alerts came from three speakers, including two
unchanged mixes. Two edited clips gave the same wrong range as their unchanged
counterpart. At the selected point, requiring mouth motion gave the same
counts as the stereo-only reading.

The cue compares short in-phase and channel-difference spectra across four
bands and estimates the contribution of room sound from their relationship.
It requires eligible stereo and an observed drop in shared sound. The visual
arm also requires located mouth motion before, during and after. Ten separate
development speakers defined the checked settings. The selected reading found
six of 20 edits with quieter ambience and three of 20 with louder ambience per
encoding in development. It flagged one ordinary control per encoding across
the two background settings. All complete development runs produced the same
selected settings. During input preflight one held recording contained two
frames and a second from the same speaker had an incompatible sound clock.
The acquisition recipe records them and selects the first qualified recording
from that speaker. The new room recordings are separate members of the same
publisher archive used for development.

The recordings come from acted speech and acoustic scenes. This study
measures staged removal localization under two measured background levels.
It does not establish detection performance for natural edits, synthetic
speech, or unfamiliar recording equipment.
[CREMA-D terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt)
and [DCASE 2013 stereo recordings](https://archive.org/details/dcase2013_scene_classification).

## Reproduce

The [development receipt](evaluation-panels/video-shared-sound-development-2026-11-17.json.gz),
[held receipt](evaluation-panels/video-shared-sound-held-2026-11-17.json.gz)
and [source inventory](evaluation-panels/video-shared-sound-sources-2026-11-17.json.gz)
pin the sources, code, picture ticks, sample intervals, and complete numerical
results. Run this from `backend/` to recalculate the held results from the
recorded measurements.

```sh
.venv/bin/python -B -S -m scripts.video_shared_sound_panel review \
  ../docs/evaluation-panels/video-shared-sound-held-2026-11-17.json.gz \
  --freeze ../docs/evaluation-panels/video-shared-sound-development-2026-11-17.json.gz
```

The source script can re-acquire previously qualified publisher video clips
from the pinned source index and license. See
`scripts.video_shared_sound_sources --help`. The measurement command takes
those clips, the publisher's stereo archive with SHA-1
`0f8088da2f45bd699bb6e39870dc8c6b624ddb96` and the pinned mouth-landmark
asset. It uses a continuous 30 fps picture clock, verifies the original
decoded pictures and writes the videos with exact sample and input digests.
See `scripts.video_shared_sound_panel measure --help`.

An [independent evidence receipt](evaluation-panels/video-shared-sound-independent-2026-11-17.json.gz)
verifies 392 local input videos and 392 parent bindings to 14 originals.
It includes case-by-case source paths, known edit ranges on the 30 fps picture
grid, and predictions for the existing mono reading and both stereo readings.
Both picture-grid and sample-grid comparisons give the held counts above.
To reproduce file verification after rendering, run
`scripts.export_video_shared_sound_review` with the held video directory,
publisher clip directory, development receipt, and a new output path. Place
the first two directories under the same local working directory. The
repository contains the receipts. The local video and audio files stay with
the evaluation workspace.
