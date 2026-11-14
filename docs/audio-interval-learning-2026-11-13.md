# Learning the location of short speech edits under competing voices

Two interval-trained checks each added three useful edit locations across
95 six-second, single-upload tests at −5 dB. Each also added nine flagged
ordinary files and between two and three flagged ordinary source groups
per test bank. The checks remain experimental. The app now shows the
coverage of the existing short-window review, including timestamps and
detected speech duration for eligible windows it skipped.

## Fit, calibration and held recordings

The existing voice model reads two overlapping four-second views of the
submitted mixture. We align its native frame responses to the source
clock and measure seven local features on 800 ms intervals every 200 ms.
They include response level, variation, contrast to nearby and outside
speech, and the share of detected speech. The check sees only the mix.
The exact edited interval is used during fitting and evaluation. No
isolated foreground, reference recording, source identity, or edit clock
enters inference. The design was informed by research that learns from
time annotations and documents loss of accuracy under unseen synthesis
and acoustic conditions. No research weights or code were imported.
[Boundary and segment learning](https://arxiv.org/abs/2609.25822) ·
[SPADE cross-condition evaluation](https://arxiv.org/abs/2609.25197) ·
[Existing voice model terms](commercial-model-audit.md)

Both seven-feature ridge heads use the same 40 earlier six-second
mixtures, including 12 edits, 24 genuine splices and four isolated
rooms. One trains on overlap at the labeled source clock. The other
also compares the exact same range of each generated splice and its
matched genuine splices. Both use one weight per clip across its
admitted ranges. Each point is set on a separate panel of 44 stock
voice and control deliveries. It allows at most one additional
ordinary group on that panel and flags no additional isolated room.
The fit and both points were sealed before held inference.

- **Randomized clock** uses 16 edits from two stock-voice generators,
  24 ordinary controls and three isolated backgrounds. Its authoring
  schedule covers the six-second clock and uses different readers and
  rooms from the stock calibration bank. [Source and terms](audio-random-clock-stock-2026-11-13.md)
- **Three generators** uses 12 edits from Chatterbox, ElevenLabs v4 and
  Kokoro, 36 ordinary controls and four isolated backgrounds. Its known
  speakers and rooms differ from the 40 clips used for fitting. This
  replay runs the current five windows on all 52 files. The earlier
  published benchmark ran only two windows, so its baseline counts differ.
  [Source and terms](audio-remote-neighbor-replication-2026-11-12.md)

Both banks had earlier project exposure. They are separate from the fit
and point-setting deliveries. The results do not establish transfer to
previously unused sources, novel generators, or natural noisy uploads.
[Fit bank and source ancestry](audio-source-isolation-diagnostic-2026-11-13.md) ·
[Point-setting inputs](audio-stock-voice-transfer-2026-11-13.md)

## Results at frozen points

An indicated range must cover at least half of the known edit and reach
0.30 temporal intersection-over-union to count as useful. A supplement
preserves every current flag and supplies a range only when the current
check did not flag. Each bank had zero isolated-background flags and
zero unassessed files with all compared routes.

| Bank and check | Useful edit locations | Misplaced edit flags | Ordinary files flagged | Ordinary source groups flagged |
| --- | ---: | ---: | ---: | ---: |
| Randomized clock, current | 1/16 | 1 | 3/24 | 2/8 |
| Randomized clock, clock head alone | 2/16 | 2 | 3/24 | 3/8 |
| Randomized clock, paired head alone | 2/16 | 3 | 3/24 | 3/8 |
| Randomized clock, current plus clock head | 3/16 | 3 | 5/24 | 4/8 |
| Randomized clock, current plus paired head | 3/16 | 4 | 5/24 | 4/8 |
| Three generators, current | 0/12 | 2 | 5/36 | 2/12 |
| Three generators, either head alone | 1/12 | 4 | 10/36 | 4/12 |
| Three generators, current plus either head | 1/12 | 5 | 12/36 | 5/12 |

The two stock gains share a single host group. The third gain was a
Kokoro replacement in a different bank. All three gains selected the
same **1.400–2.200 s** range. Training with genuine splice comparisons
did not improve the held true or ordinary counts. Either supplement
more than doubled the ordinary flags in the three-generator bank.
At this tradeoff the existing standalone speech review stays in use.
[Recorded decisions](evaluation-panels/audio-interval-learning-2026-11-13.json)

## App coverage

Eligible six-second Full reports now show the five two-second windows,
the duration of detected speech in each window, whether it was checked,
and its native source timestamps. The plain evidence view, Standard PDF
and JSON share these measurements. The coverage describes the mixed track
and carries zero weight in the AI assessment. The optional review can also
remain unassessed when the primary check is unavailable, source conditions
are unsupported, or the worker runs out of time. Those cases do not have
measured per-window coverage. Previously saved analyses retain their
original evidence schema.

Two local runs exercised checkout with simulated $9.99 USD payment.
Each queued only after confirmation. A flagged file showed the original
3.000–5.000 s waveform and all five checked windows. An ordinary file
with two seconds of leading silence reported four checked windows. The
0.000–2.000 s window showed 0.014 seconds of detected speech and was
skipped. In both cases the JSON matched the evidence API, the rendered
PDF showed all five source ranges, and the signed package replayed from
the original upload. The browser tests also cover an eligible clip with
no checked windows and an offset source clock.

An independent reviewer can rerun the speech-activity measurement for all
five windows from the submitted original. Obtain the instance public key
separately from the signed ZIP. The verifier checks the signature and the
source PCM, then reruns only the pinned speech-activity pass to compare
each reported speech count and source clock. It does not require the
voice-origin weights or a running service.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.verify_short_speech_coverage \
  "$signed_package" "$original" --public-key "$trusted_key" \
  --activity-weights models/weights/audio-upgrade/silero_vad.jit
```

## Replay

The standard-library command checks the two published stock bank
commitments, interval geometry, frozen points and held decisions.
The ledger contains scalar best responses, ranges and waveform hashes.
It contains no speech, transcripts or embeddings. Native replay requires
the retained licensed PCM banks and installed speech assets. Use a new
output directory for `$run`. The local archive contains the two stock
banks. The isolation study links to the three splice banks under
`backend/data/independent-evaluation`.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
receipt=docs/evaluation-panels/audio-interval-learning-2026-11-13.json
python3 -B -S -m scripts.audio_interval_learning review "$receipt"
for phase in freeze fit calibrate random_placement families report; do
  backend/.venv/bin/python -B -m scripts.audio_interval_learning "$phase" "$run" --archive "$stock_archive" || exit 1
done
```
