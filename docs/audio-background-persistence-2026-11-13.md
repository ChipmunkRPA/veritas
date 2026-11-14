# Do noisy speech flags persist across parts of the recording?

The current optional review found some synthetic speech after reducing loud
background sound. It also added 11 false alerts in 81 genuine noisy mixtures.
This test asks whether each flag still holds under a separate look at the
same background-reduced clip. All three probes share the voice classifier
and processed waveform, so their scores are dependent. [Earlier comparison](audio-background-safeguard-2026-11-13.md)

## Frozen method

Each branch examines an existing optional flag. It leaves the primary
voice result as recorded and can only withhold supplementary evidence.

- Temporal persistence checks the early, centered and late 1.5-second
  windows, with 0.75 seconds of overlap between neighboring windows. At
  least two must pass speech activity. Use the second-highest score.
- Disjoint persistence checks only the first and second halves. Both must
  pass speech activity, and the smaller score controls the outcome. The
  samples do not overlap. Results still depend on the same classifier and
  recording.
- Processed speech band keeps 250–3,400 Hz of the enhanced waveform using
  a fourth-order zero-phase band-pass. The full enhanced interval still
  supplies speech admission. The added inference uses the filtered result.
  The earlier source-band check used the original waveform instead.

Select each point from the 24 previously recovered noisy synthetic flags.
Use the highest observed score that retains at least 20 flags. A branch
needs enough usable window scores and must remove at least two of the 11
known genuine noisy flags in development before it can qualify for adoption.
The development replay requires the same primary assessment and supplemental
flag as before, and a whole-window logit within 0.001. Each run records its
own exact enhanced PCM digest.
All measurements are ordinal. The 1.5-second windows fall outside the
primary detector's calibrated whole-file duration range. No previously
reported percentage carries over. Temporal localization methods motivate
shorter views, although an agreement score does not determine the exact edit
position. [Research on temporal segmentation](https://arxiv.org/abs/2604.02913) · [Existing enhancement and model terms](audio-nemotron-noise-transfer-2026-11-12.md)

## Holdout and decision rule

Take 18 genuine and 18 synthetic English files from the pinned public
96 MiB RADAR evaluation prefix by deterministic hash order. These files
exclude the IDs used in the corresponding prior RADAR and enhancement
panels and share no exact source or center-excerpt samples with development.
The publisher does not identify speakers or synthesis systems for these
evaluation files, and the earlier transformations are unknown. Both
source classes use the same labeled publisher split. This prevents source
collection from serving as a proxy for class inside this holdout.
[RADAR and CC BY 4.0 terms](https://zenodo.org/records/22110815)

Keep the center three seconds and generate two matched noisy variants at
−5 dB foreground-to-added-background RMS from DEMAND street and washroom
recordings, using channel 1 of each archive. Use a single peak-safe gain
per source to keep its clean version comparable. Four background-only
excerpts from each tape enter separately. These two tapes are separate from
the development sounds and were used in other project evaluations. All
116 deliveries are exact 16 kHz PCM16 WAV clips. [DEMAND and CC BY-SA 3.0 terms](https://zenodo.org/records/1227121)

The frozen confirmation rule requires at least six current noisy synthetic
additions, retention of at least 85% overall and 75% in each background,
and retention of all clean synthetic additions. It requires at least two
fewer noisy genuine false flags and at least one removal on each tape.

## Development replay

The native input freeze matched all 204 WAV and decoded PCM commitments
from the earlier panel. The primary assessments and whole-window logits
of all 39 historical supplemental flags reproduced exactly. The
enhancement produced bit-matching PCM for 6 of 39; repeated Metal runs
on the same input could yield different PCM digests with the same
whole-window logit. The receipt records the actual enhanced PCM digest
for each new measurement.

All three corroboration probes retained 20 of the 24 noisy synthetic
flags in development. Temporal persistence dropped 3 of the 11 genuine
noisy flags, disjoint persistence dropped 2, and the processed band
dropped 4. Their points were written before any held inference.

## Confirmation results

The supplemental counts below are ordered current, temporal persistence,
disjoint persistence, and processed speech band. Each row has 18 genuine
and 18 synthetic clips. The last row pools matched variants across the
same speakers or files, so 36 deliveries per class represent 18 source
files per class.

| Condition | Primary AI flags, synthetic / genuine | Supplemental flags on synthetic | Supplemental flags on genuine |
| --- | ---: | ---: | ---: |
| Clean | 13/18 · 0/18 | 2 · 2 · 1 · 2 | 0 · 0 · 0 · 0 |
| Street | 4/18 · 1/18 | 6 · 5 · 5 · 5 | 3 · 3 · 2 · 3 |
| Washroom | 11/18 · 0/18 | 2 · 2 · 1 · 2 | 0 · 0 · 0 · 0 |
| Both noisy variants | 15/36 · 1/36 | 8 · 7 · 6 · 7 | 3 · 3 · 2 · 3 |

Temporal persistence and the processed band each dropped one noisy
synthetic flag and kept all three genuine false flags. Disjoint persistence
removed one street false flag while losing two noisy synthetic flags and
one clean synthetic flag. All eight background-only inputs were unassessed
by the primary detector and had no supplemental flags. On the noisy
variants the primary was inconclusive or unassessed on 19 synthetic and
14 genuine deliveries. The eight supplemental synthetic flags came from
these cases. Every experimental probe stays outside serving and confidence.

The native held WAV and decoded PCM hashes matched on two local
materializations, and a separately written calculation reproduced the
counts from saved responses. This test covers one publisher split, two
added background tapes, one mix level and three-second English clips.
Results cannot establish whether the same probes work on naturally noisy
recordings or on separately identified synthesis families.
[Frozen inputs and measured responses](evaluation-panels/audio-background-persistence-2026-11-13.json)

## Replay

Use the pinned RADAR archive and label paths from the [previous study](audio-background-safeguard-2026-11-13.md#replay),
its exact native input freeze as `development`, and the official DEMAND
`STRAFFIC_16k.zip` and `DWASHING_16k.zip` archives as `street` and `washroom`.
The archives, model files and audio deliveries stay local.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_background_confirmation freeze "$development" \
  --source "$inputs" --radar "$radar_dir" --llama "$llama_dir" --sounds "$sounds"
backend/.venv/bin/python -B -m scripts.audio_background_persistence freeze "$run" \
  --archive "$radar_archive" --radar "$radar_dir" --street "$street" --washroom "$washroom"
backend/.venv/bin/python -B -m scripts.audio_background_persistence fit "$run" --development "$development"
backend/.venv/bin/python -B -m scripts.audio_background_persistence score "$run"
python3 -B -S -m scripts.audio_background_persistence review \
  docs/evaluation-panels/audio-background-persistence-2026-11-13.json
```
