# Ranking short voice edits against genuine speech and background controls

Earlier tests found that a detector could mark the same time range on a
synthetic replacement and on the ordinary speech or isolated background
that accompanied it. This experiment trains the small interval detector
to compare those matched cases directly. It uses the same pinned speech
features, two-output network, speaker and room splits, and speech admission.
Research on modeling acoustic changes alongside segment content motivated
the comparison. The 16 kHz assets and model weights follow the prior
commercial-use review. [Boundary and intra-segment research](https://arxiv.org/abs/2609.25822) ·
[Prior encoder and weight terms](audio-short-edit-speech-layers-2026-11-14.md)

## Methods

Keep the existing content, boundary and paired ranking losses. For each
noisy fit set, find the highest-ranked 0.50, 0.75 or 1.00 second proposal
that covers the known synthetic edit. Compare it with the strongest
candidate from the same-reader original, genuine splice, matched
resampling, or isolated background. One variant limits control proposals
to the true edit's clock. The other considers the strongest control
proposal anywhere in each recording. All 48 matched noisy fit sets per
generator qualified for both variants. The matched controls enter fitting
and evaluation. Each inference reads one file. Separate genuine readers
set the strict and wider operating points before each held measurement.
[Prior source preparation and cutoff definitions](audio-short-edit-detector-training-2026-11-14.md)

## Held competing speech

A useful location covers at least half the known edit and reaches 0.30
temporal intersection over union. The primary panel has 24 edits, 72
ordinary foregrounds and 24 isolated backgrounds for each held generator.
It uses the same held source IDs as the previous speech-feature comparison.
The two held families use the same source and background bank.

| Wider point | Useful locations /48 | Useful without a matched control at the edit clock | Ordinary flags /144 | Isolated background flags /48 |
| --- | ---: | ---: | ---: | ---: |
| Prior frozen speech features | 2 | 1 | 20 | 4 |
| Rank against control windows at the edit clock | 1 | 1 | 0 | 2 |
| Rank against control windows anywhere in the file | 4 | 3 | 15 | 6 |

At the strict point, the two alternatives each found one useful location
with no flag on ordinary or isolated-background deliveries. The useful
locations came from different held generator families. The prior fit's one
strict useful location also appeared on a matched ordinary recording.
The primary counts use 48 deliveries across both generators, with shared
source groups retained in the numeric receipt.

## Further transfer across five generator families

The 48-file room transfer holds out three generator families from fitting.
All edits contain background speech 5 dB above the target voice. On this
panel, the whole-file ranker fitted on MeloTTS reproduced a known
ElevenLabs location at its strict point with zero flags on 33 ordinary
and four isolated-background files. It selected **3.600–4.100 s** inside
the known edit at **3.440–4.240 s**. The prior research head had found
this edit at its wider point. The other whole-file fit located a second
ElevenLabs edit at the wider point and flagged four ordinary files. The
second location was **3.200–3.700 s** in both the synthetic delivery and
its matched genuine splice. Neither version recovered a clone or Kokoro
location. [Room transfer source qualification](audio-remote-neighbor-replication-2026-11-12.md)

A second panel reused 124 exact PCM deliveries from eight MLS host groups
in two separate AMI rooms. The two generator families are Piper and KittenTTS.
It includes 16 dry, 16 equally loud and 16 louder-background edits, with
matched ordinary controls and four isolated background files. These
deliveries had earlier project exposure and were new to the fitted heads.
The source audit ruled out shared PCM and checked the recorded room and
reader bindings. On the 32 noisy edits, all four candidates found **zero**
useful locations. The current review located **8/32** with **11/48** ordinary
flags. The broadest candidate found two of 16 dry edits and flagged three
of 24 ordinary dry files. Both banks' isolated backgrounds are reported
separately. The primary panel also checks each edit against its own exact
isolated mix. [Stock voice source qualification and terms](audio-stock-voice-transfer-2026-11-13.md)

Whole-file training improved the primary panel's matched comparisons.
Its strict room result reproduced an already known location, and the
additional wider location collided with a genuine splice. The stock
transfer showed no noisy localization gain. The candidates remain in
evaluation without contributing to customer confidence.

## Independent replay

The receipts contain source hashes, sample windows, model response values,
and fitted operating points. Speech recordings and intermediate features
stay local. The portable review checks the input commitments and recomputes
calibration, useful ranges, ordinary flags, and matched-control alerts on
the same edit clock. It also compares additions and losses against the
prior frozen audio head on the room transfer. A second local CPU run
reproduced both the fitted comparison and the transfer receipt byte for byte.

```sh
export PYTHONPATH=backend
panels=docs/evaluation-panels
module=scripts.audio_competing_speech_ranking
candidate="$panels/audio-competing-speech-ranking-2026-11-15.json"
transfer="$panels/audio-competing-speech-ranking-transfer-2026-11-15.json"
python3 -B -S -m "$module" review "$panels/audio-short-edit-detector-training-2026-11-14.json" \
  "$panels/audio-short-edit-speech-layers-2026-11-14.json" "$candidate"
python3 -B -S -m "$module" review-transfer "$transfer" "$candidate"
```

Use the exact native source bank and feature cache from the [frozen feature
study](audio-short-edit-speech-layers-2026-11-14.md#independent-replay),
the pinned room roster from the [room study](audio-remote-neighbor-replication-2026-11-12.md#replay),
and the verified `freeze.json` and `signals.npy` from the stock voice study.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
module=scripts.audio_competing_speech_ranking
"$runner" -B -m "$module" fit "$sources" "$cache" "$speech_asset" "$run/fit"
OMP_NUM_THREADS=3 "$runner" -B -m "$module" transfer "$sources" "$rooms" "$stock" \
  "$encoder" "$speech_asset" "$run/fit" "$run/transfer.json"
```
