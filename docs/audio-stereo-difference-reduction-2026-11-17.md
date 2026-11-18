# Brief narrowing in stereo audio

Chipmunk Verify compares the shared sound and the differences between the
two channels in stereo PCM and AAC audio, including video soundtracks. A
strong, brief drop in channel differences can point to a section worth
listening to even when dialogue continues. The result shows the measured
time range, before/during/after signal levels, a highlighted waveform and
playable clips. The waveform labels its display gain for the difference track.
The PDF embeds the waveform and refers to the same source
timestamps and JSON evidence. The signed package can replay the measurements
and clips against the supplied original.

Moving sounds, stereo mixing and noise processing can create the same
pattern. The difference track can contain off-center speech as well as
background sound. The observation carries zero weight in the AI assessment.
The scan uses up to 30 seconds of aligned audio and shows the strongest
qualifying interval.

## Method and data

At 16 kHz, the check measures the sum and difference of the stereo channels
in overlapping 40 ms frames, with four bands between 180 and 6,800 Hz.
It compares a candidate of 180 ms to 1.2 s with 200 ms of audio on each side.
It looks for a drop across at least two bands with comparable surrounding
levels, discounts simultaneous drops in the shared signal, and requires
the shared sound to stay audible. Candidates stay within continuous
source-clock spans. Environmental changes are also used as local cues in
[audio splicing research](https://arxiv.org/abs/1411.7084).

The 352 development deliveries used four DEMAND environments and two AMI
meetings. Before measuring the held recordings, the script froze each
operating point at the greater of its preset floor and the upper empirical
98th percentile of ordinary development scores. The drop measurement used
a margin strictly greater than 10 dB for both deliveries. The separately
evaluated spectral-shape measurement used 0.6729 for PCM and 0.6675 for AAC.

Confirmation used all 100 unaltered binaural recordings from the public
DCASE scene archive, across ten environments, and 16 real excerpts from
two AMI meetings held out from development. Forty DCASE sources also
supplied matched edits and controls with the withheld AMI foreground,
at foreground-to-background levels of −5, 4, 10 and 15 dB. The rendered
edits changed only the labeled interval in the PCM source before AAC
encoding. Correct localization required at least 0.40 interval IoU.
PCM and AAC columns are two deliveries of the same underlying examples.
Results measure staged audio changes and genuine recording variation.

- The original public scene archive was released by D. Giannoulis, E. Benetos,
  D. Stowell and M. D. Plumbley under CC BY 2.0 UK. [Publisher download](https://archive.org/details/dcase2013_scene_classification)
  · [DCASE data description](https://dcase.community/challenge2013/task-acoustic-scene-classification)
- DEMAND was released by Joachim Thiemann, Nobutaka Ito and Emmanuel Vincent
  under CC BY-SA 3.0. The AMI Meeting Corpus uses CC BY 4.0.
  [DEMAND recordings](https://zenodo.org/records/1227121)
  · [AMI license and attribution](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

## Confirmation results

| Outcome | Stereo PCM | Stereo AAC at 160 kb/s |
| --- | ---: | ---: |
| Strong reduction, correct range | 27/40 | 27/40 |
| Mild reduction, correct range | 25/40 | 23/40 |
| Background difference reduced with foreground spatial signal preserved, correct range | 27/40 | 27/40 |
| Flags on unaltered acoustic scene recordings | 0/100 | 0/100 |
| Flags on unaltered stereo conversations | 1/16 | 1/16 |
| Flags across untouched mixes, speaker changes, pans and whole-mix ducks | 10/160 | 8/160 |
| Spectral check, correct range on matched-level replacements | 0/40 | 0/40 |
| Spectral check, flags on unaltered scenes and conversations | 15/116 | 13/116 |

The exact-identical-channel comparison recovered zero of the 120 staged
reductions in either delivery. For each reduction condition, four of the 40
cases lacked a comparable audio window. The same applied to 22 of the 100
unaltered acoustic scenes. The drop measurement is enabled as contextual
evidence. The spectral-shape measurement remains in the evaluation.

## Reproduce locally

The compact receipts bind delivery and source hashes, exact truth ranges,
coverage, selected proposals, development thresholds and counts. The
example sources can be regenerated from the publisher archives. DCASE
sources are the members of `scenes_stereo.zip` with publisher SHA-1
`0f8088da2f45bd699bb6e39870dc8c6b624ddb96`. AMI member SHA-256 values
are pinned to prior source manifests. The four DEMAND scenes are DLIVING,
OOFFICE, DKITCHEN and TMETRO using channels 01 and 08 from the 48 kHz
archives. Their publisher links and extracted channel hashes are in the
development receipt. Supply directories containing the extracted
`recorded-sources.json` and each matching channel subdirectory.

```sh
cd backend
panels=../docs/evaluation-panels
.venv/bin/python -B -m scripts.audio_stereo_ambience_panel review \
  "$panels/audio-stereo-ambience-development-2026-11-17.json.gz"
.venv/bin/python -B -m scripts.audio_stereo_ambience_panel review \
  "$panels/audio-stereo-ambience-held-2026-11-17.json.gz" \
  --freeze "$panels/audio-stereo-ambience-development-2026-11-17.json.gz"
.venv/bin/python -B -m scripts.audio_stereo_ambience_panel measure development \
  "$development_ami" "$fresh_run/dev" --rooms "$room_directory_1" "$room_directory_2"
.venv/bin/python -B -m scripts.audio_stereo_ambience_panel measure held \
  "$held_ami" "$fresh_run/held" --archive "$dcase_zip" \
  --freeze "$fresh_run/dev/result.json.gz" --keep-example bus04
.venv/bin/python -B -m scripts.audio_stereo_ambience_app "$fresh_run/held" "$fresh_run/app"
```

[Development receipt](evaluation-panels/audio-stereo-ambience-development-2026-11-17.json.gz)
· [Held receipt](evaluation-panels/audio-stereo-ambience-held-2026-11-17.json.gz)
· [Local checkout, report and source replay](evaluation-panels/audio-stereo-ambience-app-2026-11-17.json)
