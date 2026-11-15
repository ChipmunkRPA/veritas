# A second voice detector under competing speech

Can a different checkpoint locate short synthetic speech replacements missed
by the app's current two-second voice review? This comparison replays 136
six-second recordings with known word-level interventions. The potential
application is a supplemental local finding. The overall confidence
calibration is outside the test.

## Frozen comparison

Use the exact earlier AMI and ICSI inputs, verified by their full PCM hashes.
Eight genuine readers each supply timed word replacements from MeloTTS and
Chatterbox, corresponding resampling edits of the genuine words, and an
unchanged foreground. Each foreground group also has one isolated background
control. The AMI development panel uses two meeting captures, at 0 and −5 dB
foreground-to-background ratios. The ICSI transfer panel uses three different
meetings, all at −5 dB. The same foreground readers, edited words and
generators appear in both panels. Transfer therefore tests new meeting
backgrounds with the foregrounds fixed.
[Source preparation and terms](audio-nested-voice-window-2026-11-14.md#source-preparation)
· [ICSI excerpts](audio-icsi-overlap-intake-2026-11-14.md)

Run the independently trained Apache-2.0 checkpoint on one- and two-second
windows every half second, using its native feature extractor and requiring
detected speech over at least half of each window. Compare the one-second
response alone with the smaller of that response and a centered two-second
response. At either end, move the two-second context to keep it inside the
recording. Both lengths use the same model. In each arm, the strict threshold
is the larger of the second-highest AMI ordinary foreground group maximum
and the highest AMI background-only score. Freeze the thresholds before
scoring ICSI. The ranking returns at most one one-second range per arm.
[Pinned Apache-2.0 checkpoint](https://huggingface.co/Gustking/wav2vec2-large-xlsr-deepfake-audio-classification/tree/f7050b586236dc910d1157f430def2d0647b02b4)
· [Asset registry and voice-check terms](commercial-model-audit.md)

A useful range covers half the known edit and has at least 0.30 temporal
intersection over union. Preserve the incumbent ranges in the supplemental
comparison. A candidate merits a separate foreground-held trial if it adds
at least two useful locations across distinct readers, adds flags for at most
one ordinary reader group, and adds none on isolated backgrounds. An alert
on an edited recording that misses its edit range is counted separately.

## Results

| Background | Candidate | Useful locations /16 | Alerts outside the labeled edit | Ordinary alerts if added to current /24 | Isolated background alerts |
| --- | --- | ---: | ---: | ---: | ---: |
| AMI, 0 dB | One second | 1 | 1 | 11 → 12 | — |
| AMI, 0 dB | Both contexts | 1 | 2 | 11 → 12 | — |
| AMI, −5 dB | One second | 0 | 2 | 9 → 12 | 0/8 |
| AMI, −5 dB | Both contexts | 0 | 2 | 9 → 12 | 0/8 |
| ICSI, −5 dB | One second | 4 | 6 | 0 → 15 | 3/8 |
| ICSI, −5 dB | Both contexts | 0 | 8 | 0 → 12 | 1/8 |

The app's current two-second check alerted on 6/16 edited files in each
AMI condition and 1/16 in ICSI. None of those ranges met the localization
rule. Both candidates assessed every recording and had a qualifying
speech window overlapping the edit on all 48 staged edits. Under ICSI,
five useful one-second windows cleared the cutoff on five edited files.
The system ranked four of those windows first. At the two-context cutoff,
two useful windows cleared the cutoff, and both lost to unrelated times
in their recordings.

The four one-second ICSI locations came from both generators on two readers.
All pointed to **3.500–4.500 s**. In a diagnostic comparison after scoring,
the exact same window was the strongest flagged result in all four matched
genuine interventions, both corresponding untouched mixtures and both
isolated background excerpts. The isolated background scored higher than
the corresponding synthetic mixture in each of the four comparisons.
These apparent gains could therefore be driven by speech in the background.
Both candidates remain evaluation-only. The strongest next test is whether
the target speaker can be separated and scored while preserving the replaced
words, with the original mix and background scored as controls.
[Numeric receipt](evaluation-panels/audio-second-classifier-overlap-2026-11-14.json)

## Replay

Recount the published result with standard-library Python and the committed
parent receipts. The receipt contains per-window scalar responses and source
hashes. It carries no speech recordings, words, or model weights.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.audio_second_classifier_overlap review \
  docs/evaluation-panels/audio-second-classifier-overlap-2026-11-14.json
```

For native replay, use the publisher ranges and annotation files in the
[original source procedure](audio-nested-voice-window-2026-11-14.md#replay).
Run `prepare` for each panel with `--melo`, `--aligned` and `--earlier`.
Supply the prepared ICSI excerpt directory with `--icsi` only for the
second panel. Both preparations verify the entire published roster and
every PCM commitment. Use the pinned checkpoint snapshot directory and the
local `silero_vad.jit`; both are verified before inference.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
backend/.venv/bin/python -B -m scripts.audio_second_classifier_overlap measure \
  ami "$run/ami" "$pcm_ami" "$snapshot" "$speech_activity_weights"
backend/.venv/bin/python -B -m scripts.audio_second_classifier_overlap measure \
  icsi "$run/icsi" "$pcm_icsi" "$snapshot" "$speech_activity_weights" \
  --development "$run/ami/measured.json"
backend/.venv/bin/python -B -S -m scripts.audio_second_classifier_overlap combine \
  "$run/ami/measured.json" "$run/icsi/measured.json" "$run/receipt.json"
```

The reference run used an Apple Silicon CPU, PyTorch 2.13.0,
Transformers 5.15.1, NumPy 2.2.6 and safetensors 0.8.0. Choose an
existing fresh parent directory for `run`; the measure steps create
their own output subdirectories.
