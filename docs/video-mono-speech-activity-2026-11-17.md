# Speech-activity gaps in mono video

The app's previous speech-activity trial qualified stereo video. This
comparison tests whether those frozen settings also find short intervals
when one audio channel contains both the voice and room sound. The aim is
to locate a possible timing or sound change and present original evidence.
The method carries zero AI-confidence weight.

## Fixed comparison

The previous stereo speech-activity thresholds apply unchanged to PCM and
AAC. The candidate compares local speech activity in the original and a
noise-reduced reading. It requires independently measured speech on both
sides and ongoing mouth movement with before, during and after frames.
Both readings must place the decline on the same sample range.
[Earlier speech-activity comparison and weight licenses](video-speech-activity-gaps-2026-11-17.md)

The reservation uses 12 neutral CREMA-D speaker groups and 12 DCASE room
recordings outside all three earlier stereo studies. The held speakers
appeared in older timing studies. Their exact earlier source clips were
excluded. The new test samples one channel from each independent room take
and mixes it with the recorded foreground to make a genuine mono WAV before
video encoding. The study preserves source picture pixels and verifies their
original 30 fps presentation in each delivered file.
[CREMA-D license](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt) ·
[DCASE stereo scene archive](https://archive.org/details/dcase2013_scene_classification)

At each of two measured foreground-to-room levels (+8 and −5 dB), the
comparison uses two staged speech removals and five ordinary controls.
The ordinary controls are unchanged mixing, whole gain, a local gain dip,
local equalization, and a reduction in room sound. Clean originals and
clean staged mutes measure compatibility with the existing near-silence
review. Tests use both 16 kHz PCM and 48 kHz AAC-LC. A correct location
requires at least 0.4 temporal intersection over union on the presented
16 kHz sample clock. Both encodings share the underlying cases.

Local runtime admission requires at least four additional correct
loud-background locations across two or more speaker groups per encoding.
Up to four ordinary highlights per encoding at either background level,
across at most three speaker groups, qualify for review. The combined
policy gives the existing near-silence finding precedence and must preserve
the clean-edit hits. A successful comparison still requires matching source
timestamps and media in the browser, PDF and JSON during a local replay.

## Held results

The existing near-silence review located no removals during continuing room
sound. It located 11 of 12 clean mutes in each encoding. Giving that review
priority kept all of those locations. The following table shows the combined
policy on the same submitted files.

| Sound condition | Encoding | Correct locations | Misplaced edit ranges | Ordinary files flagged |
| --- | --- | ---: | ---: | ---: |
| Voice 5 dB below room | PCM | 2/24 | 0/24 | 0/60 |
| Voice 5 dB below room | AAC | 2/24 | 0/24 | 0/60 |
| Voice 8 dB above room | PCM | 11/24 | 1/24 | 1/60 |
| Voice 8 dB above room | AAC | 11/24 | 1/24 | 0/60 |
| Clean | PCM | 11/12 | 1/12 | 0/12 |
| Clean | AAC | 11/12 | 1/12 | 0/12 |

The two loud-background locations came from the hard and faded versions of
one speaker's recording. The check assessed 12 of 24 edited clips in PCM
and 10 of 24 in AAC at that level. The quieter level added locations across
six speakers in both encodings. The loud-background result fell short of
the preset minimum in both encodings. Mono extension stays in evaluation.

An independent comparison reproduced every reported count and verified all
384 file hashes and 12 parent hashes. FFmpeg and FFprobe separately verified
27,456 delivered frame timestamps, 858 conformed source frame hashes, and
14,643,200 mono samples against the receipts. These files use staged mutes
and separate room recordings. The results do not estimate performance on
natural edits or competing offscreen speakers.

## Second mono holdout

An additional 256 deliveries use eight CREMA-D speakers and eight DCASE
room recordings outside the first mono and earlier stereo speech-gap
comparisons. The speaker identities had appeared in other project tests.
The fixed thresholds and staged treatments stayed the same. The PCM and
AAC deliveries use the same eight foregrounds and rooms.

| Condition | Correct locations in each encoding | Edited files assessed (PCM / AAC) | Ordinary files flagged (PCM / AAC) |
| --- | ---: | ---: | ---: |
| Voice 8 dB above room | 9/16 | 16/16 · 16/16 | 0/40 · 0/40 |
| Voice 5 dB below room | 2/16 | 4/16 · 5/16 | 0/40 · 0/40 |
| Clean | 8/8 | 8/8 · 8/8 | 0/8 · 0/8 |

The near-silence comparator alone located none of the 16 noisy edits in
each condition. The combined check kept all eight clean locations per
encoding and returned zero misplaced edit ranges. The nine louder-voice
locations span five speaker groups. Both louder-room hits are from one
group. The check assessed 24 of 40 ordinary louder-room files per encoding.
The louder-room result again missed the predetermined minimum. Mono
speech activity stays in evaluation.

A separate readback rechecked all 256 file hashes, 18,752 picture
timestamps, and 10,001,056 presented mono samples. It excluded 12,976
trailing codec-padding samples. Four research witnesses reproduced the
source waveform, exact interval, and three frame anchors in both
conditions and both encodings. The research signature rejected a changed
byte. These witnesses used a test key outside the app. The portable
receipt includes source commitments and sample ranges.
[Second holdout measurements](evaluation-panels/video-mono-speech-activity-confirmation-2026-11-18.json.gz)

The held mixes use separately recorded room sound. Recent work evaluates
deliberately unsynchronized speech and mouth motion, and naturally
recorded speaker overlap with reverberation. Those conditions help
define further controls for this check.
[LASER paper](https://openaccess.thecvf.com/content/WACV2026/html/Nguyen_LASER_Lip_Landmark_Assisted_Speaker_Detection_for_Robustness_WACV_2026_paper.html) ·
[Real-World AVSE Challenge](https://real-world-avse.github.io/)

Recount the second holdout with the independent comparison tool.

```sh
review=$(mktemp -d)
panel=docs/evaluation-panels/video-mono-speech-activity-confirmation-2026-11-18.json.gz
python3 -B -S - "$panel" "$review" <<'PY'
import gzip
import json
import sys
from pathlib import Path

with gzip.open(sys.argv[1], "rt", encoding="utf-8") as stream:
    receipt = json.load(stream)
out = Path(sys.argv[2])
parts = {name: receipt[name] for name in ("manifest", "truth")}
for name, data in (parts | receipt["readings"]).items():
    (out / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")
PY
python3 -B -IS backend/scripts/independent_review.py compare-events \
  "$review/manifest.json" "$review/truth.json" "$review/near_silence.json" \
  "$review/preserve_near_silence.json" "$review/paired.json" --match 0.4 0.4
```

## Room reference and camera controls

A further comparison estimated the room sound from quiet blocks outside
each candidate. It looked for falling speech-band energy while other
bands stayed steady. Six separate development groups set the rule. Of
eight locked held groups, seven passed the existing mix renderer's
minimum room level. A separate readback verified 224 deliveries against
their original clocks and samples.

At the stronger tested room level, the combined check still located
2 of 14 edits per encoding. The room reference added no locations at
either room level and preserved all six clean near-silence locations
per encoding. In development, the room recordings supplied measurable
energy in only one of the required bands. The comparison fell short
of the frozen improvement requirement.

Three of five separately selected camera excerpts had contiguous
audio clocks. The earlier two-reading rule flagged two of these three
ordinary clips in both encodings. The room reference and near-silence
check flagged zero. This small real-world control set adds evidence
against expanding the mono check from staged mixes to customer uploads.
[Room reference and camera control results](evaluation-panels/video-mono-speech-room-floor-2026-11-18.json)

## Replay

The [reservation](evaluation-panels/video-mono-speech-activity-plan-2026-11-17.json.gz),
[publisher inventory](evaluation-panels/video-mono-speech-activity-sources-2026-11-17.json.gz)
and [held measurements](evaluation-panels/video-mono-speech-activity-held-2026-11-17.json.gz)
retain the delivery hashes, native frames, measured conditions, all estimated
intervals, and audio positions. The clean scope is recorded as `100` in the
condition field. Media stays local.

```sh
export PYTHONPATH=backend
panels=docs/evaluation-panels
python3 -B -S -m scripts.video_mono_activity_panel review \
  "$panels/video-mono-speech-activity-held-2026-11-17.json.gz" \
  "$panels/video-speech-activity-gap-development-2026-11-17.json.gz" \
  "$panels/video-mono-speech-activity-plan-2026-11-17.json.gz"
```

Rebuild the source reservation with `scripts.video_mono_activity_sources`
and obtain the selected public corpus copies with
`scripts.video_covariance_sources download`. Use the archive and source
release documented for the [previous series](video-stereo-covariance-2026-11-17.md#reproduce).
The `measure` subcommand uses those sources, the pinned local model and
picture assets, and the previous stereo freeze. Its `--help` lists inputs.

### Compare another detector

The [portable comparison](evaluation-panels/video-mono-speech-activity-independent-2026-11-17.json.gz)
contains the exact-input manifest, the intervention intervals, the native
picture and audio clocks, and predictions for the candidate, the existing
near-silence check, and the combined policy. The export rechecks each delivered
file, publisher clip and original room recording. It verifies that every
picture stream matches the conformed frames and that the conformed RGB frames
match the publisher clips. It also reconstructs the intervention from the
publisher speech and verifies that the clean mute changes only that sample
range. Audio coordinates are half-open ranges on the 16 kHz presentation
clock. Candidate picture brackets carry the frame index, presentation
timestamp on the 1/30,000 clock, and corresponding audio sample index. AAC
samples decoded beyond the presentation end are counted separately.

The prediction receipts are intended for comparing location and coverage.
An assessed file with no reported region has an empty list. The retained
near-silence measurements record a range when one was found. Coverage for its
other files is unknown, including those labeled `not_applicable` in its
portable response. The paired results count files and distinct speakers so
the two encodings and two edits per original can be interpreted in context.
Any tuning informed by these public labels needs a separate holdout.

The extract below can recount a reference response or score another detector
on the same files. Preserve each case ID and exact input hash. Use the matching
rule shown below and retain abstention reasons alongside any findings.

```sh
review=$(mktemp -d)
base=docs/evaluation-panels/video-mono-speech-activity
python3 -B -S - "$base-independent-2026-11-17.json.gz" "$review" <<'PY'
import gzip
import json
import sys
from pathlib import Path

with gzip.open(sys.argv[1], "rt", encoding="utf-8") as stream:
    receipt = json.load(stream)
directory = Path(sys.argv[2])
parts = {key: receipt[key] for key in ("manifest", "truth", "file_index")}
for key, document in (parts | receipt["readings"]).items():
    (directory / f"{key}.json").write_text(json.dumps(document), encoding="utf-8")
PY
python3 -B -IS backend/scripts/independent_review.py count-events \
  "$review/manifest.json" "$review/truth.json" "$review/preserve_near_silence.json" \
  "$review/counts.json" --match 0.4 0.4
```

To rehash the original inputs, set `local_media` to the directory with sibling
`held` and `sources` folders and set `archive` to the DCASE stereo archive. The
full export uses Python 3.11 or later, FFmpeg, and FFprobe. It compares the
originally measured files without loading model weights. The local readback
used FFmpeg 7.0.2.

```sh
python3 -B -IS backend/scripts/independent_review.py verify-inputs \
  "$review/manifest.json" "$review/file_index.json" "$review/file-checks.json" --root "$local_media"
PYTHONPATH=backend python3 -B -S -m scripts.export_video_mono_activity_review \
  "$base-held-2026-11-17.json.gz" \
  "docs/evaluation-panels/video-speech-activity-gap-development-2026-11-17.json.gz" \
  "$base-plan-2026-11-17.json.gz" "$base-sources-2026-11-17.json.gz" \
  "$archive" "$local_media" "$review/rebuilt.json.gz"
```
