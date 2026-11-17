# Short speech edits after separating stereo voices

This comparison follows the [segment-trained detector test](audio-frame-logit-localization-2026-11-17.md#separate-segment-trained-checkpoint)
that recovered zero extra edits in louder conversation. It asks two
questions. Can the same segment-trained checkpoint locate edits in
the exact room-rendered voice? Can it recover those locations when only
the stereo mix is available?
[Segment-aware speech research](https://arxiv.org/abs/2601.21925)

The exact tracks come from the 28 previously frozen two-voice scenes.
The stereo replay lowers the quieter rendered stem by another 5 dB.
This creates a nominal −10 dB two-voice condition. Its three input views
are the submitted channels averaged to mono, the existing spatial-mask
estimates, and a two-input independent vector analysis estimate. The last
view uses 24 iterative projection updates and projects the separated
tracks back to the original stereo mean. On 12 unchanged development
scenes, the 2,048-sample unwhitened variant gave the highest median
quiet-voice SI-SDR of the four separator variants tested. Using the best
matching track only for this diagnostic, held median quiet-voice SI-SDR in
the 16 unchanged scenes was −10.0, −1.0, and +0.5 dB respectively.
[Exact voices and source terms](audio-two-voice-isolation-2026-11-15.md#inputs)
· [Auxiliary-function independent vector analysis](https://arxiv.org/abs/2009.09402)

## Edit timing and ordinary alerts

The segment checkpoint gives 160 ms readings. The scan ranks ranges of one
to six bins by absolute level or by their rise relative to the stronger
of two adjacent 320 ms sections. At least half of the selected range must
contain admitted speech. The strict point uses the maximum ordinary
development speaker-pair reading for each input view. The wider point
uses the second-highest maximum. The intervals match if they cover half
of a labeled edit and reach 0.30 temporal intersection over union.

The two held cohorts contain 64 edit events and 32 ordinary mixed
deliveries. Different conditions on the same scenes reuse source tracks.
The exact-stem diagnostic measures 32 unique edited voices and 64 unique
ordinary tracks. On those exact tracks, the strict nearby-contrast scan
located 11 of 32 edits with seven ordinary flags. Six came from Chatterbox
and five from MeloTTS. The strict absolute scan located zero edits and
flagged four ordinary tracks.

| Nearby-contrast input | Correct times at the strict point | Ordinary flags at the strict point | Correct times at the wider point | Ordinary flags at the wider point |
| --- | ---: | ---: | ---: | ---: |
| Mono mix | 0/64 | 1/32 | 7/64 | 13/32 |
| Existing spatial estimate | 2/64 | 3/32 | 6/64 | 5/32 |
| Independent two-channel estimate | 4/64 | 4/32 | 7/64 | 8/32 |
| Both separated views combined | 6/64 | 5/32 | 11/64 | 11/32 |
| All three views combined | 6/64 | 5/32 | 17/64 | 17/32 |

The combined rows count the union of candidate time ranges and match each
true edit once. They are descriptive replays at the original operating
points. At the strict point, three of the four correctly timed events from
the independent estimate also favored the correct voice at that exact
interval. Neither separated method located a stock-voice edit at the
strict point. The gain is limited to one held cohort, and most edits
remain unlocated. The separator and segment checkpoint stay in evaluation.

## Recount and replay

The receipts retain input fingerprints, selected half-open intervals,
per-range scores, and acoustic comparisons. Use portable Python to
recalculate the frozen operating points and recount timings on all 140
development and held mixtures. Raw voices and frame score sequences stay
with the local evaluation inputs.
[Development result](evaluation-panels/audio-separated-segments-development-2026-11-17.json.gz)
· [Held result](evaluation-panels/audio-separated-segments-held-2026-11-17.json.gz)

```sh
export PYTHONPATH=backend
panels=docs/evaluation-panels
python3 -B -S -m scripts.audio_separated_segments verify \
  "$panels/audio-separated-segments-held-2026-11-17.json.gz" \
  --development "$panels/audio-separated-segments-development-2026-11-17.json.gz"
```

To run the original inference, use the [exact rendered stereo input recipe](audio-two-voice-components-2026-11-15.md)
and the [same verified segment checkpoint](audio-frame-logit-localization-2026-11-17.md#separate-segment-trained-checkpoint).
The local inference adapter supplies `measure(pcm)` and expects the
verified inference-only weight file as its first argument. The measured
receipt binds that adapter and checkpoint by checksum. The comparisons use
the unchanged 16 kHz clip clock, never the estimated source identity.
