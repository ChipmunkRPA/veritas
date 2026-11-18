# Periodic attenuation for short speech edits

A single-upload preprocessing test recovered one additional short edit at
both existing research operating points. The matched genuine recording
already flagged the same location. The number of edits distinguished from
their matched ordinary and room controls stayed at two of eight.

## Comparison

The filter estimates a strong, locally stable period from the submitted mono
waveform filtered to 70–1,000 Hz for analysis. It checks 48 ms frames every
12 ms. Candidate delays are 42–213 samples at 16 kHz. Each frame uses the
shortest correlation peak that reaches 0.65 and is within 0.06 of the
strongest peak. When a neighboring period is within 10%, the filter subtracts
up to 45% of the delayed signal. Weighted overlap-add preserves the sample
clock. Pitch-period subtraction can attenuate a periodic interfering sound,
but the strongest period in a mixture can also come from the target voice.
The method uses the existing detector and adds no model weights.
[Primary harmonic cancellation research](https://journals.sagepub.com/doi/10.1177/23312165211041422)

Eleven previously used Opus groups each include an edited utterance, a
matched genuine utterance, and their common room capture. The 75 known edits
come from six generator families. At least ten of the 11 groups have
simultaneous talkers in the room recording. The frozen 750 ms schedule can
reach eight of the 75 edits. Both windows in each pair must contain at least
half speech and exceed the same existing operating point. A predicted range
must cover at least half the edit and meet 0.30 temporal intersection over
union. Each prediction can match one edit.
[Original comparison, exact schedule, and source terms](audio-severe-overlap-separation-2026-11-18.md)
· [Earlier exact-foreground comparison](audio-exact-foreground-2026-11-18.md)

The filtered view used 94 of at most 96 one-thread CPU model calls. All 16
pairs passed speech admission in each edited and ordinary arm. Fourteen of
16 passed in the filtered room arm. Both points and the sample schedule
stayed fixed. The review covers these known groups and has no independent
transfer set.

## Results

| Input and operating point | Located edits out of 8 | Clear of matched controls | Ordinary alerts out of 11 | Room alerts out of 11 | Edited files flagged only at the wrong time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Submitted mixture, stricter | 2 | 2 | 2 | 0 | 1 |
| Filtered, stricter | 3 | 2 | 2 | 0 | 1 |
| Submitted mixture, wider | 3 | 2 | 2 | 0 | 0 |
| Filtered, wider | 4 | 2 | 2 | 0 | 0 |

There were no lost locations on these windows. Allowing a flag from either
view produced the same counts as filtering alone. The additional edit spans
4.053–4.473 s in one group. The untouched utterance in that group flagged the
same time in both views. The filter provided no additional control-clear
location.

After scoring, we applied the filter parameters estimated from each mixture
to the known foreground and room components independently. This measures
level changes on the reference components without using them to choose the
filter or make detection decisions. Across 75 edit clocks, the median
foreground level change was −0.12 dB, median interference change was
−0.82 dB, and median foreground-to-interference improvement was +0.69 dB.
The ratio improved on 61 intervals, including 25 by more than 1 dB. Three
worsened by more than 1 dB. Even the recovery at +2.84 dB local ratio change
coincided with an ordinary alert. The filter stays out of serving and
confidence calculation.

## Offline recount

The numeric receipt retains anonymous decision bits, per-interval level
differences, and SHA-256 commitments. It includes no sample sequences,
speaker embeddings, model files, or recording identifiers. The recount
uses the published timing and baseline ledger. Detector inference requires
the original benchmark and local model assets.
[Numeric receipt](evaluation-panels/audio-periodic-cancellation-2026-11-18.json.gz)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_audio_periodic_cancellation \
  docs/evaluation-panels/audio-periodic-cancellation-2026-11-18.json.gz
```
