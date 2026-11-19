# Segment-trained speech check on native noisy edits

The installed SAL checkpoint offers a distinct segment-trained representation
from the current two-second and 750 ms voice checks. We tested its unchanged
weights and four previously frozen operating points on 496 original recordings.
The stricter local-contrast variant found two river-masked edits that the
current checks missed, alongside six ordinary alerts. Both correct river
locations came from one host. Restaurant-masked edits remained unresolved.
This candidate stays in development.

## Frozen method and source scope

Each input is one native 12-second, 16 kHz mono PCM recording. The detector
receives the unmodified mixture. Its output has 160 ms bins. The existing
readout tests ranges of one to six bins with at least half their samples
admitted as speech. The absolute variant uses the mean bin reading. The
contrast variant subtracts the stronger of its two 320 ms neighbors. Each
variant selects at most five ranges using the existing overlap suppression.
A correct range must cover half the labeled edit and reach 0.30 temporal
intersection over union on the original sample clock.

| Historical operating point | Strict | Wider |
| --- | ---: | ---: |
| Absolute level | 10.786407470703125 | 9.697216033935547 |
| Local contrast | 6.430497348308563 | 6.051407992839813 |

These points came from the earlier development panel. Their names describe
that calibration procedure. They do not guarantee a false-alarm rate on these
recordings. This diagnostic froze the method, points and source roster before
inference. It declared no new numerical product acceptance gate and fitted
zero parameters. The app and overall confidence remain unchanged.
[Earlier calibration and results](audio-frame-logit-localization-2026-11-17.md)

The source panel contains sixteen host recordings from eight readers, with
four components connected by genuine donor splices. Each host has two stock
synthetic replacements and five ordinary variants. The latter are unchanged
speech, same-reader and different-reader splices, and two matched resampling
roundtrips. Each replacement covers 960 ms with the same boundary treatment.
The four conditions are clean, washing appliance, river and restaurant.
The designated foreground is 10 dB quieter than the added background at the
edit clock. Three physical background sessions supply the noise conditions.
There are 448 foreground deliveries and 48 isolated backgrounds.
[Native source panel and rights](audio-dense-native12-confirmation-2026-11-18.md)

All 496 deliveries are fresh to this segment model. These exact recordings
were already examined by the current speech checks and earlier-copy studies.
The generator voices were used in previous development. Reader identity
across corpora and backbone pretraining overlap remain uncertain. Conditions,
generators and ordinary variants reuse hosts. The denominators describe
deliveries and matched comparisons, with eight readers and four connected
components supplying the foreground groups.

The official checkpoint card and inference source declare MIT. The WavLM
backbone terms are CC BY-SA 3.0. The retained LibriSpeech source release is
CC BY 4.0, and DEMAND is CC BY-SA 3.0. The stock generator terms and exact
source preparation remain those documented with the source panel. Weights
and recordings stay local.
[Official checkpoint](https://huggingface.co/MaoYC/SAL)
· [Pinned inference source and terms](https://github.com/SentryMao/SAL/tree/b4e80d1d2fd98f3d8b9a85a47837ab3a73e12485)
· [Backbone terms](https://github.com/microsoft/UniSpeech/blob/main/LICENSE)
· [LibriSpeech](https://www.openslr.org/12)
· [DEMAND](https://zenodo.org/records/1227121)

## Localization and false alerts

The baseline combines the existing two-second and raw 750 ms checks. Each
condition contains 32 synthetic and 80 ordinary deliveries. The three noisy
conditions also contain sixteen isolated backgrounds each. Counts below
show correctly located synthetic files after adding the candidate and newly
flagged ordinary files. The baseline already flagged five restaurant ordinary
controls and two synthetic restaurant files at incorrect positions.

| Addition | Clean locations / 32; new ordinary / 80 | Washing locations / 32; new ordinary / 80 | River locations / 32; new ordinary / 80 | Restaurant locations / 32; new ordinary / 80 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 28; — | 8; — | 0; — | 0; — |
| Absolute level, strict | 28; 7 | 8; 1 | 0; 0 | 0; 0 |
| Absolute level, wider | 28; 11 | 8; 2 | 0; 1 | 0; 0 |
| Local contrast, strict | 32; 37 | 20; 20 | 2; 6 | 0; 0 |
| Local contrast, wider | 32; 37 | 22; 24 | 2; 8 | 0; 0 |

The stricter contrast point leaves 0, 12, 30 and 32 synthetic files without
a correct combined location in clean, washing, river and restaurant
respectively. Its candidate alone correctly locates 26, 18, 2 and 0 files.
It also flags 3, 3, 2 and 0 synthetic files entirely at incorrect locations.
The ordinary false-alert rates added by this point are 46.25%, 25%, 7.5%
and 0%. The five preexisting restaurant ordinary alerts remain.

The two correct river locations cover both generator families on one host
and one connected component. The six ordinary river alerts span two hosts
and two components. Different-reader splices account for two of those
alerts, and each of the other four ordinary roles contributes one. In clean
speech the same candidate flags fifteen different-reader splices, ten
same-reader splices, four unchanged recordings and eight resampling controls.
The findings support investigating a more selective local cue. They also
show that a response to an edit can arise from genuine speech processing.

| Condition | Synthetic with no admitted range / 32 | Ordinary with no admitted range / 80 | Isolated backgrounds with no admitted range |
| --- | ---: | ---: | ---: |
| Clean | 0 | 0 | — |
| Washing | 0 | 0 | 16 / 16 |
| River | 14 | 35 | 16 / 16 |
| Restaurant | 20 | 50 | 16 / 16 |

Every isolated background is unassessed by this speech requirement. Zero
background alerts therefore provide no evidence of assessed background
specificity. The river and restaurant synthetic unassessed rates are 43.75%
and 62.5%. The same counts apply to the absolute and contrast readouts in
this panel. Unassessed and missed locations remain separate outcomes.

## Diagnostic using the true edit clock

We separately measured the overlap-weighted segment reading at each known
960 ms edit clock, with speech admission bypassed for this diagnostic only.
Each synthetic reading is compared with all five ordinary versions of the
same host at that exact time. This uses truth to choose the evaluation
position. It measures discrimination with a supplied clock and supports no
claim of an automatic detector or calibrated confidence.

| Condition | Synthetic reading above matched ordinary reading / 160 | Synthetic files above all five ordinary versions / 32 |
| --- | ---: | ---: |
| Clean | 148 | 20 |
| Washing | 153 | 25 |
| River | 104 | 9 |
| Restaurant | 48 | 0 |

These 160 comparisons share 32 synthetic files and five ordinary versions
per host. They are correlated comparisons. Washing preserves a useful local
ordering even though the free search produces many genuine-edit alerts.
River retains a weaker cue concentrated in a small source scope. Restaurant
ordering falls below half even with the true clock supplied. Lowering a
screening point or relaxing speech admission would leave this last
representation problem unresolved.

A bounded next study can prospectively freeze an interior-versus-boundary
readout against same-reader and different-reader splices, then confirm it on
new host groups and physical noise sessions. The current results can inform
development and remain consumed evidence for any later policy selected from
them. The candidate supplies no new app behavior or overall confidence.

## Local verification and public recount

The run closed at 496 of 496 deliveries in 465.706 seconds on two CPU threads.
The complete original checkpoint matched its official digest. Its 571
inference tensors loaded strictly. An independent source implementation
replayed sixteen native hosts, 336 exact additive noisy deliveries and
79,134,720 samples outside the edit masks. The first input produced identical
readings on an immediate repeat.

A separate standard-library implementation rechecked 37,200 prediction bins
and 215,760 range admissions after closure. Its selected ranges matched the
historical float32 selector on all 1,984 case/point comparisons. Eight authored
boundary checks and 68 authored selector comparisons passed. A preflight
count assertion was corrected before protocol sealing or model calls. A
separate custody supplement pinned the unchanged historical speech-admission
asset during the run before outcomes were opened. Sources, readout, points
and labels stayed fixed.

The public receipt contains anonymous grouped decision tables, role counts,
family totals and source-group counts. It omits source identities, per-source
hashes, offsets, absolute times, response vectors, features, paths and media.
One opaque commitment binds the full private custody package. The checker
pins the named receipt, rejects extra fields and recounts intersections,
combined hits, added alerts, misses and matched-control comparisons.
[Anonymous aggregate](evaluation-panels/audio-sal-native-strong-noise-2026-11-18.json)
· [Portable checker](../backend/scripts/review_audio_sal_native_strong_noise.py)

```sh
python3 -B -S backend/scripts/review_audio_sal_native_strong_noise.py \
  docs/evaluation-panels/audio-sal-native-strong-noise-2026-11-18.json
```

The public command replays decision arithmetic with the Python standard
library. Exact native/model replay requires the retained private source
manifest, original recordings and pinned model assets. The sanitized receipt
alone cannot reconstruct the selected recordings. Repeating those sources
checks implementation parity. A new operating policy requires separately
frozen development and confirmation groups to establish fresh performance.

## Transfer to traffic and competing speech

Two contrast readouts tested whether sustained evidence inside the edit
could recover noisy locations. One measured the weakest adjacent pair of
bins against the stronger neighboring flank. The other also required
contrast inside the candidate against its first and last bins. The same
four development components fixed strict and wider points before transfer.
No weights were fitted.

The transfer used 368 retained 12-second deliveries from eight separate
MLS host and donor groups. Each condition had 32 synthetic and 80 ordinary
deliveries. Traffic and competing speech each had 16 additional isolated
background recordings. Two generator families reused settings from
development. The recordings were fresh to SAL and the new readouts.
Earlier project voice checks had seen them. Exact delivered audio had no
overlap with development. Physical identity aliases across corpora and
upstream training exposure are unresolved.

Every operating point, including the historical local contrast, found
**0 of 32** correct added locations under traffic and **0 of 32** under
competing speech. The strict persistence point added four useful clean
locations to the serving baseline of 21 of 32, with six additional ordinary
alerts out of 80. The wider persistence point found the same 25 clean
locations with 23 additional ordinary alerts. The stricter historical
contrast point flagged eight of the 16 isolated competing-speech recordings.
The two sustained readings flagged none of those 16 at either point.
Each sustained reading left 13 of 32 traffic-masked synthetic deliveries
unassessed at both points. No arm met the frozen noisy recovery gate. An
independent float32 check reproduced all 736 historical-contrast selections
across the two points. The combined confidence uses the existing checks.
