# Find speech timing changes under loud background sound

Chipmunk Verify can find a short interval where speech matches an earlier
recording at different time offsets on either side. The report shows both
offsets, how much matching sound remains inside, and each recording's own
timestamps and waveform. The browser and JSON offer both playable
excerpts. The JSON also records the source sample ranges and their hashes.

The local check needs an earlier clean copy and an analyzed copy of the
same recording. Both must be continuous 16 kHz mono PCM16 WAV or FLAC
files of 5.8 to 12 seconds, with equal numbers of decoded samples and a
verified zero start. Ordinary cuts and recordings processed in different
ways can produce timing differences. The finding contributes zero weight
to the AI assessment.

## Method and limits

Estimate the submitted recording's broad frequency profile with median
Welch spectra. Downweight the stronger frequency bands in both copies and
cross-correlate 120 ms windows every 40 ms over offsets of up to 800 ms.
The correlation used to locate corresponding samples follows the
[generalized correlation approach](https://doi.org/10.1109/TASSP.1976.1162830).
Each supporting flank needs at least three prominent matches at a stable
offset, including matches at least 160 ms apart. Require a timing change
of at least 16 ms between the flanks. Inside the candidate range, project
the source waveform on both submitted alignments. Rank the range by the
median of the stronger projection in each active window relative to the
weaker median gain from the two flanks. Require less than 38% retained
signal and suppress differences explained by broad processing.

The first prototype added flags on 28 uniformly resampled ordinary
comparisons. A separate stress set also showed why overlapping matches
alone can mistake uniform speed changes or whole-clip equalization for
local changes. The final version requires equal-duration excerpts,
independent sample support on both sides, and a measurable offset change.
Its operating point describes a waveform measurement, independent of AI
confidence.

## Local comparison

The 468 pairs reuse two earlier SINE reader panels. The same 27 labeled
splices were delivered clean or under rain, train noise, or competing
speech mixed 5 dB above the original foreground. A useful position covers
at least half the labeled span and reaches 0.30 temporal intersection
over union. Baseline includes the existing exact, acoustic, timing, and
speech-alignment comparisons.

| Cases | Earlier check | With waveform check |
| --- | ---: | ---: |
| Clean edits located | 24/27 | 25/27 |
| Edits located under rain | 0/27 | 15/27 |
| Edits located under train noise | 0/27 | 12/27 |
| Edits located under competing speech | 0/27 | 13/27 |
| Ordinary pairs flagged | 4/312 | 4/312 |
| Same-word resynthesis pairs flagged | 0/48 | 0/48 |

The waveform check displayed 43 noisy ranges, of which 40 met the overlap
rule. The other 38 noisy deliveries had no supported waveform range.
The selection criteria use two panels that were already evaluated with
other methods, including the first prototype. Their noisy conditions
reuse edits and a publisher generator. These results support a local
trial and need independent replication across generators, source devices
and real background mixes. They measure earlier-copy localization. The
test did not establish standalone AI-origin accuracy.

A separate 96-pair ordinary stress set uses eight stock voices with
varying background levels, background-only input, gain, EQ, and speed
changes with and without restored duration. It yielded zero waveform
flags. Forty-one pairs were unassessed, including 16 duration mismatches
and 16 background-only inputs. The comparison runs alongside the existing checks.

[Measurement ledger](evaluation-panels/audio-reference-waveform-2026-11-14.json)
· [SINE and background preparation](audio-aligned-speech-app-2026-11-14.md#held-comparison)
· [Stock voice preparation](audio-random-clock-stock-2026-11-13.md)

## Local use and replay

Set `VERITAS_AUDIO_REFERENCE_MATCHING_WAVEFORM=true` alongside
`VERITAS_AUDIO_REFERENCE_COMPARISON_ENABLED=true` and
`VERITAS_AUDIO_REFERENCE_CLEAN_SPEECH=true`. The waveform flag defaults to
off. In a completed retained audio analysis, open **Compare with an earlier
audio recording** and select **The earlier file is a clean copy of the same
recording**.

Three simulated USD 9.99 checkouts covered a splice with train noise, the
intact noisy comparison, and a uniformly resampled clip. No analysis job
started before payment. The submitted splice range was 2.700–3.468 s,
paired with 2.700–3.060 s from the earlier recording. The returned PDF and
JSON agreed on both ranges and the source replay verified the excerpts.
[Local checkout receipt](evaluation-panels/audio-reference-waveform-checkout-2026-11-14.json)

The numeric review reads the stored results without model weights.
For the native replay, prepare both SINE folders as before and the held
stock source folder containing `placements-freeze.json` and
`placements-native.npz`. Keep the exact pinned rain and train files.
The checkout smoke test uses the local worker's configured weights and a
Stripe simulator.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
receipt=docs/evaluation-panels/audio-reference-waveform-2026-11-14.json
python3 -B -S -m scripts.audio_reference_waveform_transfer review "$receipt"
backend/.venv/bin/python -B -m scripts.audio_reference_waveform_transfer run "$new_receipt" \
  --full "$full_sine_bank" --short "$short_sine_bank" --stock "$stock_bank" --sounds "$backgrounds"
backend/.venv/bin/python -B -m scripts.audio_aligned_speech_app "$local_app_output" \
  --bank "$full_sine_bank" --sounds "$backgrounds" --waveform
backend/.venv/bin/python -B -m scripts.compare_audio_references \
  earlier.wav submitted.wav comparison.json --verify
```
