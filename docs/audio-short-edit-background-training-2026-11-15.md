# Training short speech-edit checks with isolated background speech

The room-trained short-edit detector sometimes flagged isolated ordinary
conversation. Those recordings contributed to calibration, while the
optimizer learned from original speech, conventional edits and mixtures.
This comparison gives the optimizer the exact isolated competing speech
as a separate negative. It asks whether the detector can keep the local
edit signal while suppressing responses to the shared background.
[Earlier room experiment](audio-short-edit-room-training-2026-11-15.md) ·
[Research on noisy multi-speaker localization](https://arxiv.org/abs/2507.16220) ·
[Research on learning speech edit interiors](https://arxiv.org/abs/2601.21925)

## Frozen comparison

Start from the two pinned, centered room-trained checkpoints. Keep the
normalization, six-second clips, 64 log-Mel bands, dry and room deliveries,
−8 dB level control, and +5 dB competing speech. Each fold uses 48 fitting
prompts. The background-negative fits use the same 48 matching isolated
clips. Continue for six epochs with the same seed, minibatches and
optimizer in all three fits.

1. **Continued fit** repeats the original content, edge, and paired ranking
   objectives.
2. **Background negatives** adds negative content and transition targets on
   the isolated background.
3. **Background ranking** adds those negatives and also ranks the synthetic
   region inside its mixture above both the isolated response at the exact
   same clock and the strongest isolated half-second response anywhere in
   the clip.

Only the training procedure pairs a mixture with its separate stem.
Each held detector receives the final six-second file alone. The existing search proposes
half-second, 0.75-second and one-second regions with enough detected speech.
Each fit's stricter and wider operating points come from the 16 disjoint
ordinary calibration prompts. They use the maximum and nearest-rank 95th
percentile, respectively, with the highest isolated conversation response
as a floor. A useful prediction covers at least half the known change and
reaches 0.30 temporal intersection over union. The comparison reports
misplaced intervals, collisions at the same clock on genuine controls,
and unassessed inputs separately. These settings are for research replay.

## Matched and recorded-room results

Each PhonemeDF fold evaluates edits from the other generator with 72
ordinary and 24 isolated-background controls. Both folds examine the same
16 stock edits, 16 ordinary controls and eight background controls. The separate
recorded-room panel contains 11 synthetic edits, 33 ordinary files and
four isolated backgrounds. These source banks had prior project use and
are disjoint from fitting. No new pretrained checkpoint is used.
[Pinned source roles and permissions](audio-short-edit-room-training-2026-11-15.md#source-roles-and-delivery-conditions)

| Rank comparison, stricter point | Useful edited ranges | Ordinary flags | Isolated background flags |
| --- | ---: | ---: | ---: |
| Held Chatterbox with louder speech | 0/24 to 0/24 | 35/72 to 19/72 | 6/24 to 4/24 |
| Held MeloTTS with louder speech | 0/24 to 0/24 | 18/72 to 9/72 | 0/24 to 0/24 |
| Stock with louder speech, MeloTTS-trained fold | 3/16 to 1/16 | 5/16 to 3/16 | 2/8 to 1/8 |
| Stock with louder speech, Chatterbox-trained fold | 0/16 to 0/16 | 2/16 to 2/16 | 1/8 to 1/8 |

Each cell compares the original room fit with the background ranking fit.
The equal continuation cleared one ordinary flag on held Chatterbox and
three on held MeloTTS. At the wider point, the ranking fit found 2/24
held MeloTTS edits versus 1/24 for the original, with 31/72 ordinary flags
in both and isolated-background flags falling from 4/24 to 2/24. The
equal continuation matched those wider counts. The other held generator
did not gain a correctly timed edit. Neither fold
localized a correct edit in the separate four-room panel at either point.

## Fresh source comparison

The fresh SINE input freeze selects 18 different declared readers from
four previously unused publisher archive prefixes. It excludes the
declared readers from the earlier SINE preview, original trial and
independent-reader localization trial. Each matched original and synthetic
cut-and-paste edit has identical crop coordinates. A hash selects a
six-second crop containing the entire publisher label. Each source pair
is delivered dry and with one matched AMI room excerpt at −5 dB relative
to the room. The eight room excerpts were used in earlier SINE studies and
come from rooms separate from these fits. Together they supply eight more
isolated controls, for 80 total deliveries. Cross-corpus speaker identity
overlap and the speech-activity gate's pretraining exposure to the recordings
are unknown. Source WAVs and frame features stay local.
[Apache-2.0 publisher release](https://huggingface.co/datasets/PeacefulData/SINE#license) ·
[Synthetic splice method](https://arxiv.org/html/2501.03805#S3.SS2) ·
[Room data and source preparation](audio-sine-resynthesis-transfer-2026-11-14.md#sources-and-controls)

All eight experimental readings assessed all 18 noisy edits and all 18
noisy originals. They found no useful noisy edit location at either
point. At the wider point, the MeloTTS-trained ranking fit returned two
wrong times on the noisy edits and flagged two ordinary noisy files. Its
starting fit had no edited flags and one ordinary flag. The same ranking
fit returned two useful locations on the dry edits, matching its starting
count. The current app's short scan found zero useful locations in both
deliveries, with one ordinary flag under competing speech and none dry.
Given the known answer after inference, only **1/18** edits can fit an
admitted two-second window under the overlap rule. The narrower grid
could fit all **18/18** if supplied the correct position. Its zero noisy
locations point to a remaining response and selection problem. Both new
objectives remain in evaluation.

## Independent replay

The receipts carry source and model hashes, exact sample positions, score
maxima, calibration points and aggregate outcomes. Both commands use
standard-library Python and need neither model assets nor source media.
The two receipts share the same fitted weight and calibration commitments.
A second native intake and inference pass reproduced all 80 delivery
hashes and their eight candidate readings.
A separate fit reproduced the six weight digests, calibration and all
main and recorded-room readings. An independent recount matched the
12 frozen points for the new fits and 40 held outcome summaries.

```sh
export PYTHONPATH=backend
training=scripts.audio_short_edit_background_training
sine=scripts.audio_short_edit_background_sine
python3 -B -S -m "$training" review docs/evaluation-panels/audio-short-edit-background-training-2026-11-15.json.gz
python3 -B -S -m "$sine" review docs/evaluation-panels/audio-short-edit-background-sine-2026-11-15.json.gz
```

For native replay, restore the sealed room plan, its initial fit and the
prepared 48-recording independent-room directory. The earlier SINE
replay describes acquiring pinned 192 MiB prefixes with response headers.
Use shards **1, 4, 5 and 6** with that command and retain the original SINE
`background` directory. The fresh intake checks the publisher object
length, ETag and prefix digest. Use fresh output directories for each
phase. [Room reconstruction](audio-short-edit-room-training-2026-11-15.md#reproduce-locally) ·
[SINE inputs and bounded downloads](audio-sine-resynthesis-transfer-2026-11-14.md#replay)

```sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
runner=backend/.venv/bin/python
speech="$voice_weights/silero_vad.jit"
"$runner" -B -m "$training" fit "$room_inputs" "$initial_fit" "$speech" "$run/fit"
"$runner" -B -m "$training" measure "$room_inputs" "$run/fit" "$recorded_room_inputs" "$speech" "$run/held"
"$runner" -B -m "$training" combine "$run/fit/fit.json" "$run/held/held.json" "$run/primary.json.gz"
"$runner" -B -m "$sine" freeze "$sine_prefixes" "$sine_backgrounds" "$run/sine"
"$runner" -B -m "$sine" measure "$run/sine" "$initial_fit" "$run/fit" "$speech" "$run/fresh"
"$runner" -B -m "$sine" current "$run/sine" "$run/fresh" "$voice_weights" "$run/current.json"
"$runner" -B -m "$sine" combine "$run/fit/fit.json" "$run/fresh/readings.json" "$run/current.json" "$run/fresh.json.gz"
```
