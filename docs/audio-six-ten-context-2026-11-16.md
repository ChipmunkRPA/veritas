# Six- and ten-second contexts for short speech edits

Two new detectors were fitted to the same speech edits and matched
ordinary processing. The comparison asks whether four more seconds of
surrounding audio help identify a short edit under loud competing
speech. Both detectors read the same six-second search interval and
use the same speech admission there. The longer detector receives up
to ten seconds of source audio around that interval.

## Fitting and calibration

The preparation verifies the publisher's native WAV and alignment
archives. It preserves a one- or two-word edit of 250–600 ms within
natural ten-second host audio. The donor supplies the same words,
and its duration and level match the original phrase. A matched
ordinary control passes the original phrase through the equivalent
round trip in sample rate and the same fades. Each prompt supplies
two different generation systems, the untouched original, and the
ordinary processing control for each system. [PhonemeDF release and annotations](https://zenodo.org/records/19522979)

Sixty-four utterances from eight readers and four AMI meetings supply
the fit. Twenty-four utterances from three separate readers and two
other meetings supply the calibration. Preparation uses the same
exact background samples at each compared position. The source
word annotations confirm that the competing speech overlaps at
least half the edit. Fitting uses dry audio and alternates −5 and
−10 dB mixtures. Calibration uses both mixtures and includes the
scaled background by itself. The audit checked all 1,112 six-second
views as byte-exact excerpts of their respective ten-second views,
with 800 checks of unchanged material outside edited ranges.
[Input audit](evaluation-panels/audio-six-ten-context-input-audit-2026-11-16.json)
· [AMI terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

The same frozen speech encoder and two-output temporal head are used
for both fits. A local content loss supervises the short edit masks
and an endpoint loss supervises nearby boundaries. A matched ranking
loss compares each generated region with the corresponding ordinary
reprocessing at the same clock position. Both fits start with the
same seed and use 24 epochs. The calibration compares maximum
ordinary scores across six window widths, with isolated background
limiting the selected points. Recent work motivates matched
ordinary splices and separate checks of frame and event transfer.
[Temporal anchors and ordinary splice controls](https://arxiv.org/abs/2610.04479)
· [Segment-aware speech localization](https://arxiv.org/abs/2601.21925)
· [Encoder pin and weight terms](audio-wavlm-v13-protocol.md)

## Retained transfer

The retained panel has 240 native deliveries from 48 recordings.
The earlier application reading is kept intact. Both fitted heads
seek ranges within the center six seconds. At each condition, 77
known intervals are fully inside that scope, including 33 of at
most 600 ms. Evaluating the candidates on that shared scope keeps
search extent and speech admission fixed. It evaluates the effect
of giving the head more context. We count one-to-one matches that
cover at least half the known edit with at least 0.30 intersection
over union. [Retained input source and earlier serving results](audio-two-second-long-review-2026-11-16.md)

At the wider research point the six-second fit added one accurate
location on clean audio and one at −5 dB. The ten-second fit added
one at −5 dB. The additions at −5 dB were on separate short edits.
At the strict point the six-second fit preserved its one −5 dB
addition with the same ordinary and background alert counts.

| Condition | Current found / 77 | Add six-second fit | Add ten-second fit | Add either fit | Ordinary alerts, current → either / 24 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 42 | 43 | 42 | 43 | 0 → 1 |
| Competing speech, −5 dB | 21 | 22 | 22 | 23 | 5 → 8 |
| Competing speech, −10 dB | 11 | 11 | 11 | 11 | 2 → 7 |

The wider combined rule adds one isolated-background alert at each
noise level, from one to two of 48. At −10 dB neither fit adds a
correct location. Requiring the two candidates to agree adds zero
correct locations across the retained panel. The evaluated heads
are outside the app. [Numeric retained comparison](evaluation-panels/audio-six-ten-context-retained-2026-11-16.json.gz)

## Fresh comparison of additions

A separate panel uses 24 new edited utterances across six generator
families, 24 ordinary recordings, six other meetings, and 46
foreground readers. Every utterance and clean waveform has a new
digest. Some speakers appeared in earlier test sets. The speakers
and meetings used to fit these heads are excluded. Each condition
contains 73 known edit intervals within the shared central six
seconds, 24 unmodified recordings and 24 ordinarily processed
controls. Each noisy condition also contains 48 exact isolated
background deliveries.
[LlamaPartialSpoof terms and labels](https://zenodo.org/records/14214149)
· [LibriSpeech source terms](https://www.openslr.org/12/)

Both heads used their previously frozen points on all 312
deliveries. At least one head proposed a range on
37 deliveries. The app's current rule was run on all 37. On
every other delivery, the combined locations equal the app's
existing locations by construction. A replay of this procedure
on the full 240-delivery retained result reproduced the changes
using the 25 deliveries that could change. The fresh table
reports incremental changes. Absolute current recall and
ordinary-alert counts need the separate full baseline replay.
[Frozen fresh roster and per-delivery changes](evaluation-panels/audio-six-ten-context-fresh-delta-2026-11-16.json.gz)

| Condition at the wider point | Add six seconds, edit locations / untouched alerts / processed alerts | Add ten seconds, edit locations / untouched alerts / processed alerts | Add either, edit locations / untouched alerts / processed alerts |
| --- | ---: | ---: | ---: |
| Clean | 0 / 0 / 0 | 0 / 2 / 2 | 0 / 2 / 2 |
| Competing speech, −5 dB | 3 / 0 / 0 | 2 / 3 / 3 | 5 / 3 / 3 |
| Competing speech, −10 dB | 1 / 1 / 1 | 2 / 3 / 3 | 2 / 3 / 3 |

The six-second fit adds three correct −5 dB locations across three
source groups. At the stronger setting it adds one useful location
with one additional alert in each ordinary category. The strict
six-second point adds one −5 dB location with zero added alerts
there. Requiring agreement recovers one location at −10 dB at the
same additional ordinary counts as the wider six-second fit.
No variant adds an isolated-background alert. The six-second fit
is the stronger candidate for a further test using the original
utterances as matched controls. The app continues using its
existing checks while that comparison is prepared.

## Same-utterance controls

The original LibriTTS dev-clean release contains the corresponding
recording IDs for all 24 edited inputs. A waveform check verifies
22 of those associations. It requires at least two distinct
0.4-second excerpts outside the known edited regions to be found
in the same order in the original. Each excerpt must correlate
at 0.94 or higher and beat every position outside its immediate
neighborhood by at least 0.08. The two unverified associations
remain outside the matched panel. [Original LibriTTS sources and terms](https://www.openslr.org/60/)
· [How the edited corpus used LibriTTS](https://arxiv.org/html/2409.14743#S2.SS2)

Fourteen of the verified pairs cover at least nine seconds on a
shared clock and have a complete labeled edit in the central six
seconds. They span all six generator families and 42 labeled
edits in each condition. For each pair, the verified excerpt
nearest the middle sets a time offset and loudness ratio. Both
recordings retain their natural relative timing. Each competing
speech condition uses the same samples and amplitude in both
members of the pair. An independent subtraction verified all
56 edited or ordinary mixtures against their isolated backgrounds.
The frozen panel contains 112 deliveries. Scoring uses the same
fit and operating points. [Source matches and input commitments](evaluation-panels/audio-six-ten-context-matched-inputs-2026-11-16.json.gz)
· [Publisher checksum](https://www.openslr.org/resources/60/checksum.md5)

## Replay

The portable verification reproduces the per-width and whole-input
points and recounts all event outcomes. The export supports another
detector's selected sample intervals on the same manifest. Its
truth includes confirmed intervals wholly inside the shared scope.
Each case includes the original sample count and input SHA-256.

```sh
export PYTHONPATH=backend
receipt=docs/evaluation-panels/audio-six-ten-context-retained-2026-11-16.json.gz
python3 -B -S -m scripts.audio_long_context_review "$receipt"
python3 -B -S -m scripts.audio_long_context_review "$receipt" --export "$review"
python3 -B -S -m scripts.independent_review count-events "$review/manifest.json" \
  "$review/truth.json" "$review/current_plus_ten.json" "$recount" --match 0.30 0.50
delta=docs/evaluation-panels/audio-six-ten-context-fresh-delta-2026-11-16.json.gz
python3 -B -S -m scripts.audio_long_context_review "$delta" --incremental
```

The matched controls use the LibriTTS dev-clean archive with
SHA-256 `da0864e1bd26debed35da8a869dd5c04dfc27682921936de7cff9c8a254dbe1a`.
Restore the original fresh panel and its pinned AMI excerpts first.
The builder saves exact source and mixture hashes.

```sh
runner=backend/.venv/bin/python
"$runner" -B -m scripts.audio_matched_originals audit "$archive" "$fresh" "$work/originals.json"
"$runner" -B -m scripts.audio_matched_originals prepare "$archive" "$fresh" "$work/originals.json" "$rooms" "$work/matched"
"$runner" -B -m scripts.audio_long_context_transfer measure "$work/matched" "$work/measurements" "$encoder" "$vad"
"$runner" -B -m scripts.audio_long_context_transfer delta "$fit" "$points" "$work/matched" "$work/measurements" "$weights" "$work/delta.json"
```
