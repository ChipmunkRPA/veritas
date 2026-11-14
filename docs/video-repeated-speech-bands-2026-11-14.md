# Repeated soundtrack fragments across three speech bands

Can the single-track video check find a short repeated speech fragment when
background noise obscures part of the frequency range? We compared the earlier
two-band method with a three-band method and with a method requiring the same
two bands out of three throughout the repetition. The alternatives found one
additional clean AAC edit and no additional noisy edits. Both remain offline.

## Comparison

The three new bands were 220–900, 900–2,100, and 2,100–4,100 Hz. All arms
looked for adjacent or nearly adjacent repetitions using the earlier 60 ms
window, 10 ms step and waveform-correlation floor. The requirements for
speech activity and different moving mouth images stayed the same. In the
two-of-three arm, a particular pair of bands had to support the whole run.
No additional weights were used. The existing face geometry model has a
pinned MIT license. A sound can recur naturally, and timing alignment is
distinct from whether the visible mouth articulates the words correctly.
[Earlier repeat method](video-adjacent-sound-transfer-2026-11-14.md)
· [Frequency weighting under noise](https://www.isca-archive.org/interspeech_2021/salvati21_interspeech.pdf)
· [Timing and articulation study](https://arxiv.org/abs/2610.09223)
· [Face weights and license](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

The comparison uses four previously tested excerpts. The held part uses
ten newly prepared 2.52-second excerpts from five separate published
recordings, two excerpts per recording. The five recordings had prior
project use at other offsets. Each excerpt supplied a local 200 or 280 ms
shift in both directions. Controls covered untouched recordings, local
gain, filtering or noise, whole-track delay, an added repeated bed, and
moving the picture and sound together. Synchronized controls were rendered
and decoded on their own picture clock. Competing speech came from a
different recording in the same source split. Its standard deviation, and
that of the train background, was set 5 dB above the untouched foreground
before peak-safe rendering. Each scene was delivered in PCM and in actual
48 kbps AAC. The same source and edit are reused across carriers and
noise conditions. [Public conversations and license terms](video-lipsync-repeat-fresh-actors-2026-11-13.md#fixed-method-and-sources)
· [Held recording sources](evaluation-panels/video-local-alignment-2026-11-14.json)
· [Train background and terms](audio-focused-window-transfer-2026-11-12.md#frozen-comparison)

Each repeated interval needed at least 0.50 temporal intersection over
union with its own labeled repeat. The decision required three additional
noisy AAC locations across at least two recordings, at most one new
control flag per carrier, and zero new synchronized or whole-track flags.

## Held results

Both alternatives produced the same counts. The table keeps all the earlier
adjacent-repeat findings and adds the new ones.

| Held delivery | Earlier useful | With either candidate | Controls flagged, earlier → combined |
| --- | ---: | ---: | ---: |
| Clean PCM | 1/20 | 4/20 | 0 → 1 of 90 |
| Clean AAC | 3/20 | 4/20 | 1 → 1 of 90 |
| PCM with competing speech | 0/20 | 0/20 | 0 → 0 of 30 |
| AAC with competing speech | 0/20 | 0/20 | 0 → 0 of 30 |
| PCM with train noise | 0/20 | 0/20 | 0 → 0 of 30 |
| AAC with train noise | 0/20 | 0/20 | 0 → 0 of 30 |

Four of 20 edits in each held delivery group had insufficient mouth crops
and remain in the denominator. The added clean AAC location shows
**1.430–1.710 s** and **1.710–1.990 s**, within 10 ms of the known repeat
intervals. The same measurements also found a real repeat in a separate
PCM control where the picture and sound were moved together. This was a
false lip-sync lead. The two copies alone cannot establish which changed,
the full shift interval, or AI origin. The earlier repeat comparison itself
remains experimental and is distinct from the production timing check.

The analysis view and Standard PDF now also report the distinct duration
covered by measured timing windows and the distinct duration scheduled.
Overlapping windows count once. The existing source-clock windows in the
JSON contain the intervals and dispositions needed to recalculate both
durations.

## Replay

The public receipt includes measurements, decoded frame commitments and
sample positions. The standard-library command checks and recounts all
588 deliveries. Native replay uses the pinned public recordings and train
file listed in the source receipts. Raw audio and picture files stay local.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
receipt=docs/evaluation-panels/video-repeated-speech-bands-2026-11-14.json
python3 -B -S -m scripts.video_repeated_speech_band_transfer review "$receipt"
python3 -B -S -m scripts.video_repeated_speech_band_transfer freeze "$fresh_freeze"
backend/.venv/bin/python -B -m scripts.video_repeated_speech_band_transfer run "$fresh_receipt" \
  --freeze "$fresh_freeze" --comparison "$comparison_sources" --held "$held_sources" --sounds "$sounds"
cmp "$receipt" "$fresh_receipt"
```
