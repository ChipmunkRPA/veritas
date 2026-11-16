# Separate entry and exit cues for speech edits under competing voices

Earlier short-speech checks reported overlapping intervals on some
synthetic edits, genuine edits and background-only recordings. This
comparison tests whether a short region has different evidence at the
two ends. It measures signed changes in speech features against a
matched control that keeps only the magnitude of those changes.
Independent boundary and content predictions are motivated by recent
localization research. The study uses existing pinned assets and its
own training and scoring code. [Prior matched-room results](audio-clean-teacher-transfer-2026-11-16.md)
· [Boundary and segment research](https://arxiv.org/html/2609.25822v1#S2.SS1)
· [Genuine-splice controls](https://arxiv.org/abs/2610.04479)

## Frozen comparison

Two training directions each use 48 prompts from six declared
LibriSpeech readers, with dry and louder competing-speech deliveries.
Each fit uses one synthetic generator and its corresponding resampling
control. Untouched recordings and genuine splices also enter both
fits. The other generator is held out. The same frozen encoder and
projection supply the representations for all arms. [Source and asset terms](audio-short-edit-speech-layers-2026-11-14.md)

Both heads share the same temporal content network and number of
parameters. They each learn one reading for speech content, one for
entry and one for exit. Each boundary label covers 30 ms on either
side of its authored join. One head reads the signed change in
features between 40 ms before and after the candidate point. The other
reads the absolute change across that same pair of points. The content
network reads the original features in both experiments.

Score proposals of 0.50, 0.75 and 1.00 seconds every 100 ms. Speech
activity must cover at least half of the interval. Average the content
reading over the interval and add 0.15 times either the mean or the
smaller of the two endpoint readings. For each endpoint, take its
strongest appropriate reading within 40 ms. Each arm and readout is
calibrated independently from the same two held-aside genuine readers
and their isolated backgrounds. The wider point uses the 95th
percentile of the ordinary foreground readings, bounded below by the
highest isolated background reading. The stricter point uses the
greatest ordinary or isolated reading. All four fitted heads and both
points per readout were sealed before evaluation.

A useful location covers at least half of a labeled edit and reaches
0.30 temporal intersection over union. An alert at a different time
is tracked separately from a correct location. When testing a fallback,
the app's original highlighted range has priority. The receipt also
contains the candidate-only results. The internal operating points are
comparison settings for these data.

## Held recordings

The first 460 submissions come from previously measured source panels
outside the present fit and calibration. They include the two
cross-generator ICSI comparisons, three groups of SINE readers at
different noise levels, and rendered two-voice mixtures from Aachen,
Huddersfield and a separate BBC room. Each previously published parent
establishes the clock, source roles and recording terms. Previously
exposed audio is retained as a comparison, with the untouched recordings
and isolated backgrounds where supplied. [SINE and primary source roles](audio-local-speech-normality-2026-11-16.md)
· [Stereo room controls](audio-stereo-channel-replication-2026-11-15.md)
· [BBC room controls](audio-clean-teacher-transfer-2026-11-16.md#held-and-fresh-outcomes)

The fresh comparison uses the first 192 MiB of two previously unused
SINE archive shards at the pinned publisher revision. Nine eligible
readers remained after excluding the named earlier source panels and
primary prompt hashes. Each reader contributes a single
publisher-labeled edit. The corresponding ordinary speech also has a
round-trip resampling control at the same sample range. The three
versions share the identical AMI background at each simulated
interference level and a single gain that keeps the isolated background
and mixtures within the PCM range. These rooms have prior project
exposure. Test the dry recording, a mixture 5 dB below the competing
speech and one 10 dB below it. Replay the exact transmitted background
at each level as a fourth input. The confirmation comprises 99
submissions. [SINE publisher terms](https://huggingface.co/datasets/PeacefulData/SINE)
· [AMI recording terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

## Held results

The table gives the signed-change readout with the mean endpoint term
at the wider point. The rows outside ICSI use the Chatterbox fit. The
full receipt gives both fits, both operating points and
all four comparisons on every panel.

| Panel | Useful ranges, current → fallback | Ordinary alerts, current → fallback |
| --- | ---: | ---: |
| ICSI, opposite-generator tests summed | 4/48 → 5/48 | 64/144 → 64/144 |
| Aachen simulated room | 1/18 → 1/18 | 0/12 → 0/12 |
| Huddersfield simulated room | 2/18 → 6/18 | 1/12 → 1/12 |
| BBC room, both independent source cohorts | 2/36 → 2/36 | 0/24 → 0/24 |
| Earlier SINE reader comparisons | 1/77 → 1/77 | 21/77 → 21/77 |

The Huddersfield additions come from two scenes, with one single-voice
edit and one two-voice edit in each. Three of the four had already
been recovered by the aligned clean teacher on exactly those deliveries.
The unsigned comparison finds the same four useful Huddersfield
deliveries and adds a misplaced range. Both signed endpoint readouts
find one additional useful ICSI range, **4.200–4.700 s**, with no
matched ordinary or isolated-background flag at that clock. The
unsigned readouts also find an additional ICSI range, along with more
ordinary and isolated-background alerts. The second fitted model
gives no added useful range on the simulated rooms. The primary table
counts ordinary observations in both fit directions, where some
controls share the same source. [Held comparison receipt](evaluation-panels/audio-edit-direction-held-2026-11-16.json.gz)

Every fresh input had an eligible window in both the current scan and
the candidate checks. Both fits and all four variants found zero useful
locations on the nine fresh edits at each noise level and in the dry
recordings. On the stronger **−10 dB** mixes, the current check and both
signed fallbacks each flagged six of 18 ordinary recordings. At
**−5 dB**, the current check flagged eight of 18. The signed Chatterbox
fit increased that count to ten and added one misplaced flag on an
edited file. The other fit retained eight ordinary flags. All variants
flagged zero of 18 exact isolated background submissions.

Before applying a cutoff, the signed mean from the MeloTTS fit found
the right interval in five of nine dry edits and in two of nine at
each interference level. The Chatterbox fit's top proposal matched
three dry edits and zero at both noisy levels. The localized gains in
one simulated room and one older ICSI example did not transfer to the
new readers. The candidate stays in evaluation.
[Fresh paired receipt](evaluation-panels/audio-edit-direction-fresh-2026-11-16.json.gz)

## Independent replay

The portable checks verify cutoffs and time overlap in all 416 outcome
cells. Their output matched on Python 3.12 and 3.14. A second fit
regenerated each head and its calibration byte for byte. A fresh
rebuild reproduced all 99 PCM files. The separate media check then
verified the hashes, unchanged samples outside the ordinary edit, and
the foreground-to-background ratios of the test mixtures. The public
receipts contain time ranges, scalar readings and source commitments.
The audio, speech representations and fitted weights stay local.

```sh
export PYTHONPATH=backend
module=scripts.audio_edit_direction_transfer
panels=docs/evaluation-panels
for split in held fresh; do
  receipt="$panels/audio-edit-direction-$split-2026-11-16.json.gz"
  python3 -B -S -m "$module" review "$receipt"
  python3 -B -S -m scripts.audit_audio_edit_direction "$receipt"
done
```

For native replay, use the [clean-teacher source preparation](audio-clean-teacher-transfer-2026-11-16.md#replay).
Use its `phoneme`, `sine`, `additional` and `stereo` inputs and its
feature cache at `$feature_cache`. Its two follow-up media directories
are `$recent_sine` and `$recent_bbc`, with their exact scoring receipts
at `$recent_sine_receipt` and `$recent_bbc_receipt`.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=2
runner=backend/.venv/bin/python
mixed="$panels/audio-local-speech-normality-2026-11-16.json.gz"
weights=models/weights/audio-upgrade
encoder=models/research/wavlm-v13
"$runner" -B -m "$module" fit "$phoneme" "$feature_cache" "$weights/silero_vad.jit" \
  "$mixed" "$recent_sine" "$recent_bbc" "$run/fit"
"$runner" -B -m "$module" score "$phoneme" "$feature_cache" "$weights/silero_vad.jit" \
  "$encoder" "$mixed" "$sine" "$additional" "$stereo" "$recent_sine" "$recent_sine_receipt" \
  "$recent_bbc" "$recent_bbc_receipt" "$run/fit" "$run/held.json.gz"
```

Use the [SINE bounded source procedure](audio-sine-resynthesis-transfer-2026-11-14.md)
for shards `30` and `31` to prepare `$prefix_cache`. Use a fresh
output folder alongside the existing fit and media.

```sh
fresh_module=scripts.audio_edit_direction_fresh
"$runner" -B -m "$fresh_module" build "$phoneme" "$recent_sine" "$additional" "$prefix_cache" "$run/fit" "$run/fresh-media"
"$runner" -B -m "$fresh_module" measure "$run/fresh-media" "$run/fit" "$weights" "$encoder" "$run/fresh.json.gz"
python3 -B -S -m scripts.audit_audio_edit_direction "$run/fresh.json.gz" --media "$run/fresh-media"
```
