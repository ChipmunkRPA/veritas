# Testing short changes to words against the recorded mouth movements

The existing timing comparison can miss a word replacement even when audio
and video still start together. We tested whether a sequence of mouth
shapes or the detail visible inside the mouth crop could locate the
replaced word. None of the candidates located an additional replacement on
the separate 19-clip panel. The existing check remains in the application.

## Method

The fit used 84 synchronized takes from 14 speakers. It compared the
recorded mouth shapes with the corresponding speech sounds and with
signals from 240–640 ms elsewhere in the same recording. A ridge fit
used five groups of speech-sound estimates and seven mouth measurements.
Separate comparisons added motion or a grid of brightness and color
readings from the native mouth crop. The local check searched for a
mismatch bracketed by consistent speech and mouth movements on both
sides. Both operating points were fixed from 16 recordings of eight
different speakers before scoring the held files.
[Contrastive timing research](https://arxiv.org/html/2610.09223v1) ·
[Lip-focused consistency research](https://arxiv.org/abs/2610.30872) ·
[Apache-2.0 speech checkpoint](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft/tree/ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4) ·
[Landmark asset lineage](video-bilabial-mouth-transfer-2026-11-14.md)

Each scored clip retains its original picture stream. A labeled edit
uses a recording of the same speaker saying a different letter or digit.
The native comparison confirms exact audio samples outside the replaced
range. Controls include unchanged speech, gain adjustment, same-word
replacement and two full-track delays. These are staged speech edits
drawn from the CC BY 4.0 Lombard Grid corpus. Generalization to generated
speech or generated mouth motion remains unmeasured here.
[Lombard Grid corpus and terms](https://zenodo.org/records/3736465)

## Results

The earlier held panel uses 34 clips from 17 speakers. The separate
panel uses 19 clips from 12 speakers. Neither panel supplied fit
or operating-point examples. Each panel repeats the same clips in
lossless PCM and in AAC, so the two deliveries share source groups.
An audio interval counts as useful at intersection over union of 0.40
or more. Each entry shows the existing check followed by the existing
check plus the shape comparison selected on the separate calibration
speakers.

| Panel | Useful locations | Ordinary controls flagged | Same-word controls flagged | Whole-track delays flagged |
| --- | ---: | ---: | ---: | ---: |
| 34 clips, PCM | 7 → 9 of 34 | 3 → 5 of 68 | 3 → 4 of 34 | 3 → 8 of 68 |
| 34 clips, 16 kHz AAC | 10 → 11 of 34 | 3 → 5 of 68 | 3 → 7 of 34 | 3 → 7 of 68 |
| 19 separate clips, PCM | 1 → 1 of 19 | 11 → 11 of 38 | 4 → 4 of 19 | 8 → 10 of 38 |
| 19 separate clips, 48 kHz AAC | 0 → 0 of 19 | 11 → 11 of 38 | 4 → 4 of 19 | 8 → 10 of 38 |

Adding mouth-crop texture also recovered zero extra locations on the
19-clip panel. The selected shape check added a misplaced range at
each of its two deliveries and two ordinary whole-track delay flags
per delivery. Earlier counts and all alternate operating points can
be recalculated from the frozen numeric receipts.
[PCM mouth shape and motion](evaluation-panels/video-lip-correspondence-pcm-2026-11-16.json.gz) ·
[AAC mouth shape and motion](evaluation-panels/video-lip-correspondence-aac-2026-11-16.json.gz) ·
[PCM mouth-crop comparison](evaluation-panels/video-lip-correspondence-texture-pcm-2026-11-16.json.gz) ·
[AAC mouth-crop comparison](evaluation-panels/video-lip-correspondence-texture-aac-2026-11-16.json.gz) ·
[Separate panel results](evaluation-panels/video-lip-correspondence-transfer-2026-11-16.json.gz)

## Rebuild the playable benchmark

The separate panel yields 228 MOV files on demand from the two publisher
archives, six audio treatments at two deliveries for each source clip.
The native replay checked all 228 submitted files and their 228 source
bindings. It verified the compressed picture stream, every picture
timestamp and the presented audio samples. An independent reader can
match predictions on the original picture grid using the normal
evaluation contract. The frame-grid match agrees with the exact audio
metric here, with one correct PCM location and zero AAC locations
for both methods. This is a replay of labeled inputs. Use separate
holdout files when selecting a new method.
[Source roster](evaluation-panels/video-lip-correspondence-sources-2026-11-16.json) ·
[File commitments, predictions and event counts](evaluation-panels/video-lip-correspondence-playable-2026-11-16.json.gz)

Save the publisher's audio and front-camera ZIP files as
`lombardgrid_audio.zip` and `lombardgrid_front.zip` in `$sources`. The
build used FFmpeg 7.0.2 on macOS. The portable
event recount uses Python's standard library and needs no media.
Keep the reconstructed videos and dense phone and mouth measurements
in the reviewer’s local workspace. The repository carries the recipes
and numeric commitments. The benchmark has received local replay and
is available for external reproduction.

```sh
export PYTHONPATH=backend
panels=docs/evaluation-panels
transfer="$panels/video-lip-correspondence-transfer-2026-11-16.json.gz"
python3 -B -S -m scripts.video_lip_crop_correspondence verify "$transfer"
backend/.venv/bin/python -B -m scripts.export_video_lip_correspondence prepare \
  "$panels/video-lip-correspondence-sources-2026-11-16.json" "$transfer" \
  "$sources" "$run"
python3 -B -S -m scripts.independent_review verify-inputs \
  "$run/manifest.json" "$run/file-index.json" "$run/rehashed.json" --root "$run"
python3 -B -S -m scripts.independent_review count-events \
  "$run/manifest.json" "$run/truth.json" "$run/candidate.json" "$run/recount.json" \
  --match .40 .40
```
