# A second speech check under loud backgrounds

The frozen second opinion added one synthetic detection and zero genuine
false flags beyond the existing options on the English environmental-noise
panel. Adding it to the primary check under competing voices recovered
three synthetic clips and falsely flagged six more genuine clips. It stays
outside the analysis workflow and confidence scoring.

## What was compared

An earlier study on noisy Turkish speech added 18 detections across
two generators with no added false flags. It tested foreground-to-background
ratios of 0 and +5 dB. This follow-up keeps the same English
four-second fit, diagnostic points, and original three-second sample
reader. It uses a second speech representation combined with the primary
response. The fit penalizes changes in score between clean and noisy
versions of the same training utterance. Each evaluated delivery uses
only its own unprocessed, speech-admitted samples. This is a related
consistency idea to the time-frequency study below. Neither research
paper's code, data, nor weights were used.
[Prior Turkish result](audio-matched-background-score-v250-result.md) ·
[English fitting and diagnostic points](audio-paired-consistency-v223-method.md) ·
[Time-frequency consistency research](https://arxiv.org/abs/2607.17761) ·
[Speech and background benchmark methodology](https://arxiv.org/abs/2608.09593)

- **Environmental sound.** The 116 existing deliveries contain 18 genuine
  and 18 synthetic English files, with one clean and two −5 dB deliveries
  per source file, plus eight isolated excerpts. The street and washroom
  recordings come from DEMAND. RADAR does not identify these clips'
  synthesis systems or speakers. The local runner reopened every WAV and
  replayed every primary direction. The existing background-reduced
  flags came from the exact earlier measurement ledger. The second
  opinion processed 30 previously inconclusive deliveries.
  [Source rights and input preparation](audio-background-persistence-2026-11-13.md)
- **Competing speech.** The 77 archived deliveries include 16 genuine
  clips from eight readers, nine synthetic clips in three generator
  families, and two isolated background excerpts. Each foreground has
  clean, mostly single competing voice, and overlapping competing voices
  conditions. Both mixes use −5 dB foreground to background. The same
  room excerpt recurs within each scene. The *genuine foreground* label
  refers to the designated foreground, and the test offers no mechanism
  to identify which voice drives the classifier. The replay reopened
  every exact 48,000-sample input and bound it to its previously recorded
  primary response. The second opinion assessed 60 deliveries.
  [Original source and room qualification](audio-independent-background-transfer-2026-11-11.md#source-recovery-and-separate-comparison) ·
  [Generator provenance and license terms](audio-short-edit-source-plan-v169.md)

The speech checkpoint has an MIT publisher declaration. The fixed second
encoder uses the commercially permitted CC BY-SA 3.0 terms pinned in its
existing evaluation. Model assets, fitted parameters, original audio,
voice features, and individual scalar scores stay local. The evaluation
publishes anonymous direction decisions and overall commitments.
[Primary checkpoint audit](commercial-model-audit.md) ·
[Second encoder and runtime audit](audio-wavlm-three-second-v226-method.md)

## Recorded results

Each pair below is synthetic AI flags / genuine false AI flags. The
denominators include unassessed files. On the environmental panel the
last column adds the paired check to the primary and optional
background-reduced check. For competing speech, it adds the paired
check to the primary only.

| Condition | Planned synthetic / genuine | Primary | Primary + background reduction | Primary + paired | Union of available checks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Environmental, clean | 18 / 18 | 13 / 0 | 15 / 0 | 14 / 0 | 15 / 0 |
| Street, −5 dB | 18 / 18 | 4 / 1 | 10 / 4 | 5 / 1 | 10 / 4 |
| Washroom, −5 dB | 18 / 18 | 11 / 0 | 13 / 0 | 13 / 0 | 14 / 0 |
| Competing speech, clean | 9 / 16 | 7 / 0 | — | 7 / 0 | 7 / 0 |
| Mostly one competing voice, −5 dB | 9 / 16 | 4 / 3 | — | 5 / 6 | 5 / 6 |
| Overlapping competing voices, −5 dB | 9 / 16 | 5 / 12 | — | 7 / 15 | 7 / 15 |

On environmental noise the two optional methods together flagged 24 of
36 synthetic deliveries and four of 36 genuine deliveries. The existing
combination flagged 23 and four. As a replacement for background
reduction, the paired check would have found 18 with one false flag.
It also leaves primary-unassessed inputs without a second opinion. Its
one net gain was below the frozen requirement of two source files.

Under competing speech, the three recovered clips came from the
ElevenLabs and Kokoro families. Six distinct genuine utterance groups
gained a false flag, above the allowed one. Treating an original-leaning
paired score as a reason to withhold a primary AI flag removed zero
false flags at the fixed point. No isolated background received an AI
flag in either bank. Both panels reused previously evaluated recordings.
The results describe three-second decisions and do not measure exact
edit timestamps or naturally noisy upload frequencies.

The speech assessment and the flagged background-reduced review now tell
users that overlap can prevent attribution to a specific speaker. The
browser, PDF and structured customer assessment use the same limitation.

## Replay

The public reviewer recounts all 193 decisions, shared source groups,
and generator coverage from the anonymous ledger. Run it without
downloading media or loading a checkpoint. It verifies the reported
decisions from the ledger. Native inference requires the frozen inputs
and the original privately fitted head.
[Public decision ledger](evaluation-panels/audio-paired-loud-noise-2026-11-13.json)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_audio_paired_loud_noise \
  docs/evaluation-panels/audio-paired-loud-noise-2026-11-13.json
```

The [environmental source replay](audio-background-persistence-2026-11-13.md#replay)
and [competing-voice source replay](audio-independent-background-transfer-2026-11-11.md)
describe how the original input banks were made. Set `environmental`
and `speech` to new paths, `wav_bank` to the environmental run containing
`wav/`, and `pcm_bank` to the competing speech bank containing `pcm/`,
`roster.json`, `freeze.json`, and `baseline.json`.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze baseline score; do
  backend/.venv/bin/python -B -m scripts.audio_paired_loud_background \
    "$phase" "$environmental" --source "$wav_bank" || exit 1
done
for phase in freeze score; do
  backend/.venv/bin/python -B -m scripts.audio_paired_competing_speech \
    "$phase" "$speech" --archive "$pcm_bank" || exit 1
done
python3 -B -S -m scripts.review_audio_paired_loud_noise "$new_receipt" \
  --from-native "$environmental/receipt.json" "$speech/receipt.json"
```
