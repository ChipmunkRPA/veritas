# Compare separate stereo channels under competing speech

The local earlier-copy review admits twelve-second mono AAC. The previous
twelve-second stereo trial combined both channels and doubled the ordinary
flags relative to mono. This follow-up keeps the same waveform comparison
and tests whether reading both channels separately improves the result.
[Earlier AAC results](audio-clean-speech-aac-2026-11-15.md#twelve-second-mono-aac-under-competing-speech)

## Method

The retained AMI and ICSI banks provide 272 twelve-second pairs across 32
source groups. In each pair, the earlier mono recording contains the known
source speech and the submitted recording mixes in another voice. The base
mixture uses competing speech 5 dB louder than the foreground. Known changes
include synthetic replacements from four generator families and ordinary
speech substitutions. Other cases include unchanged speech, resynthesis of
the same words, a background-only half, or a wrong or half-matching reference.

The study recreates the retained stereo AAC deliveries with a scaled and
slightly delayed second channel. A second layout keeps the same submitted
speech on one channel and adds a different recorded voice on the other.
Before encoding the second layout, the contaminated channel receives 0.35
of the submitted mix plus 1.20 of the added speech at matched whole-clip
RMS. It is scaled down if needed to avoid clipping. The contaminated side
alternates by source group. Both layouts use controlled stereo mixes of
recorded speech. Transfer to natural multichannel recordings remains untested.

The decoder requires a twelve-second, continuous 48 kHz AAC-LC stereo
presentation and preserves the native left and right sample sequences as
well as the FFmpeg downmix. It analyzes the common 16 kHz prefix and drops
decoded padding beyond the original presentation end. The existing waveform
rule runs independently on each view. One policy requires both channel
ranges to share half their duration. A second policy uses the channel whose
reference match is supported and leads by at least 0.04 correlation in each
six-second half. Without a supported lead, it uses the channel agreement
policy. A useful range covers at least half the known changed interval and
achieves at least 0.30 temporal overlap relative to the union of the two ranges.
[Reference-based stereo interference research](https://arxiv.org/abs/2606.18564)
· [Two-microphone coherence research](https://pmc.ncbi.nlm.nih.gov/articles/PMC3246289/)

## Results

| Stereo layout and comparison | Correct synthetic ranges | Correct genuine-splice ranges | Ordinary clips flagged | Wrong or half-matching references admitted |
| --- | ---: | ---: | ---: | ---: |
| Retained stereo, downmix | 32/64 | 19/40 | 12/80 | 0/64 |
| Retained stereo, channel agreement | 31/64 | 18/40 | 0/80 | 0/64 |
| Retained stereo, stronger reference match | 31/64 | 18/40 | 3/80 | 0/64 |
| Added speech on one channel, downmix | 32/64 | 19/40 | 15/80 | 64/64 |
| Added speech on one channel, channel agreement | 2/64 | 1/40 | 1/80 | 0/64 |
| Added speech on one channel, stronger reference match | 32/64 | 19/40 | 6/80 | 32/64 |

On the uneven stereo mixes, selecting the channel with stronger reference
support preserved all correct ranges and removed three misplaced ranges
on edited examples. The wrong-reference controls included the exact
inserted background voice as a reference. The downmix selected four false
ranges among half-matching controls. The separate-channel rule produced
zero wrong-reference ranges, but it admitted all 32 exact-background
references for comparison. The rule identifies matching speech and cannot
tell whether that speech is the voice the user intended to examine. Every
check declined comparison on all 24 background-only target halves in each
layout. [Scalar outcome receipt](evaluation-panels/audio-clean-speech-stereo-long-2026-11-15.json.gz)

After scoring, we tested whether both halves of the weaker channel also
showed some support for the reference. On the uneven layouts, points of
0.05, 0.10 and 0.15 each removed the 32 background-reference admissions
and kept all 32 useful synthetic ranges, 19 useful genuine-splice ranges
and six ordinary flags. Raising it to 0.20 lost one useful synthetic range,
and 0.25 lost 20. This probe used the same data and the added background
was intentionally confined to one channel before AAC encoding. Real
microphone channels can share the same background. The stereo check stays
experimental pending a separate test of real stereo recordings and of
references present in both channels at different levels. The comparison
measures changes against an earlier recording and supplies no AI-origin
score.

## Replay

The receipt contains input fingerprints, exact range ticks, scalar reference
measurements and outcomes. The reviewer checks the source roster and every
retained encoded file against the earlier receipt. The native mode rebuilds
both AAC versions, decodes the samples and verifies the previous downmix
outcomes before evaluating the alternatives. It uses no model weights.

```sh
export PYTHONPATH=backend
module=scripts.audio_clean_speech_stereo_long
receipt=docs/evaluation-panels/audio-clean-speech-stereo-long-2026-11-15.json.gz
python3 -B -S -m "$module" "$receipt"
python3 -B -S -m "$module" "$receipt" --diagnostic
```

For native replay, use FFmpeg 7.0.2 and the two pinned PCM banks from the
[earlier AAC study](audio-clean-speech-aac-2026-11-15.md#local-app-and-replay).

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m "$module" "$run/stereo.json.gz" \
  --icsi "$icsi" --ami "$ami"
```
