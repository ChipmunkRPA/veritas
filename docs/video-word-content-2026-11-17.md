# Separate word readings for short video edits

The current lip-sync check can miss a substitution that preserves speech cadence. This trial reads a spoken digit from the soundtrack and separately estimates the visible digit from the sequence of mouth crops. An admitted mismatch supplies the estimated audio interval and the nearby original frames. It measures speech content disagreement. An editorial dub, a timing error, or a wrong word reading can produce it.

The experiment follows research that separates speech content and articulation from coarse synchronization. The visible and audible readers are fitted locally on genuine recordings from licensed sources. A separate trial used a general English speech model to obtain timestamps. It returned usable digit tokens on one of 13 unprompted development recordings, so the benchmark uses the source-trained acoustic reader. [Speech content consistency study](https://arxiv.org/abs/2610.30872) · [Timing and articulation study](https://arxiv.org/abs/2610.09223) · [English speech checkpoint and licence](https://huggingface.co/openai/whisper-tiny.en) · [Lombard Grid terms](https://zenodo.org/records/3736465)

## Fitting and selection

- The visual model learns a time-normalized sequence of 12 mouth crops across each word. It fits 336 original Lombard Grid recordings from 14 speakers, with publisher-aligned words for supervision. A previously fitted local visual encoder supplies the initial mouth features. Its fifteenth epoch reads 11 of 13 independently timed development digits correctly when supplied their published boundaries.
- The separate soundtrack model extracts log-mel features at 10 ms spacing. It learns the digit identity and its position from 1,162 genuine streams from the same 14 fit speakers. Its eighth epoch was the first to read and locate all 13 original development digits. The case inference receives only delivered audio and mouth pixels. It receives no publisher digit, timestamp or transcript.
- One arm compares the mouth reading at the independently estimated audio time. The other also requires the same visible digit at both nearby boundaries and checks whether an offset up to 240 ms would support the audible digit. The internal cutoffs come from the largest control response per development speaker, with the second-largest setting strict exceedance, independently for PCM and AAC. The seven development speakers are outside fitting and the new evaluation.

## Held comparison

The held panel has 24 previously unused utterances from 24 distinct speakers and a different, duration-matched donor digit for each. The new original and donor media were reserved by a deterministic source rule before measuring either model on them. Both audio deliveries preserve the same original pictures and source times. The controls include unchanged speech, gain adjustment, a same-digit replacement and whole-track delays of ±200 ms. A correct new read must match both labeled digits and overlap at least 0.40 of the union with the known edit range.

| Comparison | Different-digit edits located | Untouched or gain flags | Same-digit flags | Whole-track delay flags |
|---|---:|---:|---:|---:|
| Current PCM timing check | 0/24 | 4/48 | 5/24 | 7/48 |
| PCM timing plus one word window | 1/24 | 4/48 | 6/24 | 8/48 |
| PCM timing plus nearby-time check | 2/24 | 6/48 | 8/24 | 12/48 |
| Current AAC timing check | 0/24 | 4/48 | 4/24 | 5/48 |
| AAC timing plus one word window | 1/24 | 4/48 | 5/24 | 6/48 |
| AAC timing plus nearby-time check | 2/24 | 7/48 | 8/24 | 11/48 |

Both deliveries locate the same two edits in the nearby-time arm and only one in the stricter arm. The acoustic reading alone correctly identifies and locates 14 PCM donor digits and 13 AAC donor digits. The joint check assesses six PCM edits and seven AAC edits. The nearby-time arm adds one AAC highlight in the right area with an incorrect visual digit. The benchmark uses staged replacements with genuine recordings and a scripted digit vocabulary. The observed gain is limited to this panel, and the candidate stays in evaluation.

The audio support rule excluded eight of the 14 correct PCM digit readings and seven of the 13 correct AAC readings before testing the corresponding mouth crops. These exclusions isolate the weakest step for the next trial.

The native replay rebuilt and verified all 288 videos. It recovered the exact picture bitstream, picture timestamps and presented audio samples. The stricter arm estimates the same edit at 1.590–2.080 s in PCM and 1.590–2.070 s in AAC. Both estimates select native frames [39, 50). Frames 38, 44 and 50 show the before, during and after context. [Native readback](evaluation-panels/video-word-content-native-2026-11-17.json.gz)

[Held source plan](evaluation-panels/video-word-content-sources-2026-11-17.json.gz) · [Frozen development results](evaluation-panels/video-word-content-selection-2026-11-17.json.gz) · [Held decisions](evaluation-panels/video-word-content-held-2026-11-17.json.gz) · [Fit history](evaluation-panels/video-word-content-fit-2026-11-17.json.gz)

## Reproduce

Recount the decisions with standard-library Python from the repository root. The code checks the held operating points, model commitments, case clock bounds, and separation from the development speakers.

```bash
export PYTHONPATH=backend
panels=docs/evaluation-panels
selection="$panels/video-word-content-selection-2026-11-17.json.gz"
held="$panels/video-word-content-held-2026-11-17.json.gz"
python3 -B -S -m app.bench.video_word_content "$held" --freeze "$selection"
```

For local inference, obtain the pinned [Lombard Grid inputs](video-neural-phone-clock-2026-11-14.md#independent-replay) and the [earlier mouth crop setup and fit](video-mouth-patch-speech-2026-11-17.md#reproduce). The source plan selects the moment and fixed media pair for each test. These commands rebuild the two readers with the local fitting data and evaluate new files with the same comparison. The acoustic source list commits to every publisher member used for fitting.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
runner=backend/.venv/bin/python
run=/path/to/new-evaluation-folder
sources=/path/to/licensed-lombard-grid-archives
assets=/path/to/local-lip-timing-assets
base_fit=/path/to/earlier-mouth-model-fit
prior_plan="$panels/video-mouth-patch-sources-2026-11-17.json.gz"
new_plan="$panels/video-word-content-sources-2026-11-17.json.gz"
mkdir -p "$run"
```

The capture commands use the same landmarker and phone checkpoint bundle as the earlier study. Set `assets` to that approved directory. To rebuild the baseline receipts directly, adapt the published plans for the existing timing runner.

```bash
"$runner" -B - "$prior_plan" "$new_plan" "$run" <<'PY'
import sys
from pathlib import Path
from scripts import video_lip_sequence_panel as timing

prior, fresh, folder = map(Path, sys.argv[1:])
for source, split, measured in ((prior, "calibration", "calibration"), (fresh, "fresh", "held")):
    plan = timing.panel.read(source)
    rows = [dict(row, split=measured) for row in plan["rows"] if row["split"] == split]
    timing.panel.save(folder / (measured + "-clock-plan.json.gz"), dict(
        schema=timing.SCHEMA, method=timing.STUDY_METHOD, rows=rows,
        counts={measured: {"recordings": len(rows)}},
    ))
PY
dev_clock="$run/calibration-clock.json.gz"
new_clock="$run/held-clock.json.gz"
new_crop="$run/held-mouth-crops"
fit_crop="$run/fit-mouth-crops"
dev_crop="$run/dev-mouth-crops"
"$runner" -B -m scripts.video_lip_sequence_panel capture "$run/calibration-clock-plan.json.gz" "$sources" "$assets" calibration "$dev_clock"
"$runner" -B -m scripts.video_lip_sequence_panel capture "$run/held-clock-plan.json.gz" "$sources" "$assets" held "$new_clock"
"$runner" -B -m scripts.video_visual_speech_train capture "$prior_plan" "$sources" "$assets" "$fit_crop" --phase fit
"$runner" -B -m scripts.video_visual_speech_train capture "$prior_plan" "$sources" "$assets" "$dev_crop" --phase evaluation
"$runner" -B -m scripts.video_visual_speech_train capture "$new_plan" "$sources" "$assets" "$new_crop" --phase evaluation
```

Refit the two local readers, using the directory from the earlier mouth model fit for `base_fit`.

```bash
"$runner" -B -m scripts.video_word_content_model visual-inputs "$prior_plan" "$sources" "$fit_crop" "$dev_crop" "$run"
"$runner" -B -m scripts.video_word_content_model fit-visual "$run" "$base_fit/visual-speech.safetensors"
"$runner" -B -m scripts.video_word_content_model audio-inputs "$prior_plan" "$sources" "$run"
"$runner" -B -m scripts.video_word_content_model fit-audio "$run"
"$runner" -B -m scripts.video_word_content_transfer "$run" "$sources" "$prior_plan" "$dev_crop" "$dev_clock" "$run/selection.json.gz"
"$runner" -B -m scripts.video_word_content_transfer "$run" "$sources" "$new_plan" "$new_crop" "$new_clock" "$run/held.json.gz" \
  --phase fresh --freeze "$run/selection.json.gz"
"$runner" -B -m scripts.export_video_word_content "$new_plan" "$run/selection.json.gz" "$run/held.json.gz" "$sources" "$run/native"
```

Local media, feature arrays and weights stay on the machine doing the work. [Acoustic training sources](evaluation-panels/video-word-content-audio-sources-2026-11-17.json.gz)
