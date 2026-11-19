# Recurrent-background view for standalone speech localization

A single-upload background estimate added **zero useful edit locations** on
124 controlled deliveries. It made the synthetic river and restaurant
mixtures assessable, while both conditions still yielded zero correct
locations. This candidate stays in development. Standalone scoring and
customer confidence remain unchanged.

## Frozen comparison

The method estimates recurrent sound from the submitted twelve-second
recording. A 1,024-sample Hann transform advances by 256 samples. For each
frame, eight cosine-similar magnitude frames at least two seconds away
supply a per-frequency median. The estimate is clamped to the current
magnitude. A squared residual-to-background mask, with a 0.1 amplitude
floor, retains the original complex phase and sample count. This is an
independent NumPy/SciPy implementation inspired by recurrent separation
research. [Author's methodology](https://users.cs.northwestern.edu/~zra446/repet.html)

The existing model, internal point, speech requirement and confirmed
two-second window rule run separately on the original and processed
waveforms. The combined arm retains every original range. The method uses
the uploaded recording alone. Earlier copies, isolated backgrounds and
edit clocks enter source verification and outcome measurement only.
[Installed asset review](commercial-model-audit.md)

Four real LibriSpeech foreground recordings from four readers each supply
two 960 ms synthetic inserts, using retained Piper and Kitten voices. The
five ordinary counterparts are untouched speech, same-reader and
different-reader splices, and two local resampling controls. Every version
appears clean and with each of three real background sessions at −10 dB
relative to the original target interval. Four excerpts per background
also enter alone. These are **eight synthetic events repeated across four
conditions**, giving 32 synthetic, 80 ordinary and 12 background-only
deliveries. The ordinary donors extend the source ancestry.

The method and roster were frozen before standalone voice inference.
Foregrounds had prior exposure to the separate earlier-copy comparator.
The generator assets and background sessions had earlier project exposure.
Pretraining overlap is unknown. LibriSpeech carries CC BY 4.0 and DEMAND
backgrounds carry CC BY-SA 3.0. Exact source custody follows the preceding
native-input study. [LibriSpeech](https://www.openslr.org/12/) ·
[DEMAND](https://zenodo.org/records/1227121) ·
[Native-source qualification](audio-dense-native12-confirmation-2026-11-18.md)

## Results

Each cell compares the original scan with the combined arm. A location
must cover half the planted edit and have at least 0.30 temporal IoU.
Misses include unassessed files. Ordinary and isolated flags are counted
per delivery.

| Condition | Located / 8 | Missed / 8 | Synthetic unassessed / 8 | Ordinary flags / 20 | Ordinary unassessed / 20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 2 → 2 | 6 → 6 | 0 → 0 | 0 → 0 | 0 → 0 |
| Washing machine | 1 → 1 | 7 → 7 | 0 → 0 | 0 → 0 | 0 → 0 |
| River | 0 → 0 | 8 → 8 | 4 → 0 | 0 → 0 | 10 → 1 |
| Restaurant | 0 → 0 | 8 → 8 | 4 → 0 | 5 → 5 | 10 → 0 |

All twelve isolated-background deliveries had zero flags. Ten remained
unassessed after combination. The two flagged restaurant synthetic files
had ranges entirely outside their edits. Every useful primary location
came from Kitten. The clean Piper inserts were missed in all four cases.

The five ordinary restaurant alerts and two off-target synthetic alerts
came from one foreground recording. All seven shared the same false range,
with identical submitted PCM there and no overlap with any planted edit.
This source-bound check locates the mistaken evidence without assigning
it to a particular voice or background component.

The processed view alone reduced restaurant ordinary flags from five to
one but lost one of the two clean locations. Using it as a blanket veto
would discard useful evidence. The predeclared requirement of four added
noisy locations across three foreground readers and both generators failed.
Its ordinary, background and matched-control budgets passed.

A separately declared retrospective cutoff sweep used the closed readings.
At additional ordinary-file budgets of zero, one, two and four per
condition, with zero added background flags, its best shared point added
one washing-machine location. River and restaurant stayed at zero. This
within-panel diagnostic supplies no calibrated serving point.

## Verification and next direction

A separate implementation reproduced all 124 processed waveforms byte for
byte, checked 248 original/modified clocks and verified 16,957,440 PCM
samples outside labeled edits. A standard-library recount checked all 124
closed rows and 3,145 admitted window-clock bindings. Inference used 1,269
unique classifier forwards on two CPU threads in 593.65 seconds, with zero
weight changes. Duplicate window bindings exceed unique forwards because
overlapping contexts reuse the same samples.

Improved speech admission did not produce edit discrimination. The next
standalone comparison should train or calibrate segment evidence against
matched genuine edits and stronger backgrounds, with generator families
reported separately. The four-reader pilot and three reused background
sessions support this direction and provide limited coverage of real
recordings. The broader [reader holdout](audio-short-window-reader-holdout-2026-11-18.md)
also found no locations under its two strong-background conditions.

The public package contains aggregate counts and whole-study commitments.
Audio, identities, per-case clocks and response vectors stay private.
Its reviewer checks aggregate arithmetic. Original-input and individual
outcome replay were completed locally.
[Scalar receipt](evaluation-panels/audio-standalone-recurrent-view-2026-11-18.json)

```sh
python3 -B -S backend/scripts/review_audio_standalone_recurrent_view.py \
  docs/evaluation-panels/audio-standalone-recurrent-view-2026-11-18.json
```
