# Training short voice-edit checks on two generators

The earlier short-edit fits each learned from one type of synthetic voice.
This comparison tests whether exposure to a second generator improves the
ability to locate changed words when a louder conversation is present. The
October component study found that detector responses can transfer poorly to
unseen synthetic speech even when background-event classification works.
That result motivates holding out voice mechanisms and testing them with
matched ordinary speech and isolated room recordings.
[October component study](https://arxiv.org/abs/2610.17617)

## Fixed comparison

The two single-generator fits each use 48 prompts from six readers. The
balanced fit takes four prompts per generator per reader and keeps the same
48 prompts and training steps. The full fit takes both synthetic renderings
per prompt and doubles the number of training examples. Every fit uses the
same temporal network, frozen speech representation, training procedure and
500, 750, or 1,000 ms proposals. Genuine splices and matched resampling
controls measure sensitivity to ordinary editing. The native timeline and
speech gate stay fixed throughout. [Source split and detector method](audio-short-edit-detector-training-2026-11-14.md)
· [Encoder weight terms and pins](audio-wavlm-v13-protocol.md)

Sixteen separate ordinary prompt groups from two declared readers set both
operating points. For each group take the maximum over the dry and noisy
untouched, genuine-splice and two resampled inputs. The strict point uses the
largest of those group maxima. The wider point uses the nearest-rank 90th
percentile. Both use the largest isolated-background response as a floor.
The scores and points are research settings with no authorship-probability
interpretation. A useful proposed region covers at least half of the known
edit with at least 0.30 temporal intersection over union. The fallback
retains any interval already flagged by the current two-second review.

## Retained recordings

The test replayed 559 previously measured deliveries, including 224 edited
files, 275 ordinary controls and 60 recordings of isolated background. The
test includes held Chatterbox and MeloTTS readers, SINE edits, and Piper and
Kitten inserts rendered in measured rooms. The two generators used for
training appear in the PhonemeDF evaluation with distinct declared readers.
The encodes, mixtures and ordinary controls for one source share content.
[Retained input preparation and terms](audio-edit-boundary-direction-2026-11-16.md#held-recordings)

| Strict fallback | Useful edit locations /224 | Useful with no matched ordinary alert at that clock | Ordinary alerts /275 | Isolated alerts /60 | Wrong edit locations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current two-second review | 10 | 6 | 78 | 0 | 55 |
| MeloTTS only | 10 | 6 | 81 | 0 | 57 |
| Chatterbox only | 16 | 10 | 90 | 2 | 58 |
| Balanced fit | 16 | 12 | 84 | 0 | 58 |
| Both renderings | 20 | 11 | 88 | 0 | 60 |

Three deliveries were unassessed in each strict comparison. The balanced
fit adds three useful locations in Huddersfield and one in Aachen. It also
recovers one dry SINE edit and one PhonemeDF edit. The full fit adds six
useful Huddersfield locations, with four of its eight total useful locations
clear of a matched ordinary alert there. Both fits leave the nine most recent
SINE edits at each competing-speech level unlocalized. Replaying the wider
full fit raises useful detections from 10 to 24 and ordinary alerts from 78
to 103, with three added isolated-background alerts. The strict comparisons
show a better tradeoff on these recordings. [Retained numeric receipt](evaluation-panels/audio-joint-generator-retained-2026-11-16.json.gz)

## Separate voice-conversion test

The independent input preparation read the first 640 MiB of each of three
published PhonemeDF audio archives. It matched the original, TTS and voice
conversion recordings against the complete published alignment archive.
The initial 320 MiB contained no eligible new reader. The larger prefix yields
eight readers outside all fitting, calibration and PhonemeDF assessment roles
of the previous study. These are declared reader identities. Earlier project
exposure and pretraining overlap elsewhere remain possible. The source files,
selected word tiers and exact delivery clocks are recorded by hash.
[Publisher data and CC BY 4.0 terms](https://zenodo.org/records/19522979)
· [Earlier same-word source qualification](audio-phoneme-short-edit-intake-2026-11-13.md)

Each reader supplies one six-second recording with two same-word edits and
two ordinary round-trip resampling controls at the same 0.69–0.98 second
interval. Each source appears dry and with the same recorded competition at
nominal −5 dB or −10 dB. An additional input at each mix level contains the
exact transmitted background. There are 136 deliveries across five retained
AMI recordings. The actual foreground-to-background ratio within the edited
words ranges from −9.82 to −0.73 dB in the nominal −5 dB mix and from −14.82
to −5.73 dB in the nominal −10 dB mix. Calibration comes entirely from the
earlier source groups. [AMI release terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

| Strict fallback | Dry useful /16 | −5 dB useful /16 | −10 dB useful /16 | Ordinary alerts /72 | Isolated alerts /16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current two-second review | 12 | 1 | 2 | 15 | 0 |
| MeloTTS only | 12 | 1 | 2 | 15 | 0 |
| Chatterbox only | 13 | 1 | 2 | 15 | 0 |
| Balanced fit | 13 | 1 | 2 | 15 | 0 |
| Both renderings | 12 | 2 | 3 | 15 | 0 |

Every delivery was assessed. The strict fit trained on both renderings finds
one additional voice-conversion edit at both noise levels. It selects
**2.800–3.300 s** inside the known **2.645–3.355 s** edit. Neither ordinary
resampling nor the isolated background flags at that clock. The wider point
adds another useful location at −5 dB and flags two isolated backgrounds.
The eight-reader result qualifies the strict fit for further local trial.
The new models remain in evaluation. The noisy gains cover one additional
reader in this test, and the earlier severe SINE misses remain.
[Voice-conversion receipt](evaluation-panels/audio-joint-generator-conversion-2026-11-16.json.gz)

## Replay and compare another detector

The numeric receipts carry exact sample positions, study decisions and
source and weight hashes. Audio, frame representations and fitted weights
stay local. The source manifest can also be exported to the general
evaluation format with exact locations and counts, ready for comparison
with another detector's sample ranges. The public comparison tool reports
misses, misplaced and unmatched proposals, and unassessed inputs.
Two local fits reproduced all weights and operating points byte for byte.
A second media preparation rebuilt the 136 deliveries, and a separate
sample audit verified the untouched outer samples and both noise levels.
The generic event counter agreed with the native comparison for all five
strict fallback columns, including ordinary and isolated alerts.

```sh
export PYTHONPATH=backend
panels=docs/evaluation-panels
for split in retained conversion; do
  python3 -B -S -m scripts.audio_joint_generator_transfer review \
    "$panels/audio-joint-generator-$split-2026-11-16.json.gz"
done
python3 -B -S -m scripts.audio_joint_generator_vc export \
  "$panels/audio-joint-generator-conversion-2026-11-16.json.gz" "$output"
python3 -B -S -m scripts.independent_review count-events \
  "$output/manifest.json" "$output/truth.json" "$output/current.json" "$output/current-recount.json" \
  --match 0.30 0.50
```

Each case uses the delivered 16 kHz sample clock and an exclusive end
sample. `current.json` is an example of the response format. Use the exact
case ID, delivery hash, status and selected samples for another method.
Run the same command with that response file to count its event matches
against the fixed sample intervals. This interface needs standard Python.

For native replay, prepare the earlier [fit and held inputs](audio-edit-boundary-direction-2026-11-16.md#independent-replay).
The `assess --help` arguments refer to the same source trees. Reuse the six
room excerpts from the clean-teacher transfer's new SINE preparation as
`$backgrounds`. Restore the bounded publisher ranges with the [same-word
source recipe](audio-phoneme-short-edit-intake-2026-11-13.md#replay), extend
the three audio caches to 448 MiB with 8 MiB ranges, and save the 448–640 MiB
extension below `$extension/<archive-origin>/` as `<start-byte>.range`
with the original response header at `<start-byte>.headers`. The extension
uses 16 MiB ranges. The download endpoint is the linked Zenodo record's
`files/PhonemeDF_<archive-origin>.tar.gz` URL. Use its `download=1` query
with the exact range in `part`, and send the matching HTTP Range header.
The runner verifies prefix range lengths, publisher file sizes, archived
prefix hashes and the alignment archive's published MD5.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
weights=models/weights/audio-upgrade
encoder=models/research/wavlm-v13
"$runner" -B -m scripts.audio_joint_generator_transfer fit "$run/fit" \
  --data "$phoneme" --cache "$feature_cache" --speech "$weights/silero_vad.jit" \
  --anchor "$panels/audio-edit-direction-held-2026-11-16.json.gz" "$panels/audio-edit-direction-fresh-2026-11-16.json.gz"
"$runner" -B -m scripts.audio_joint_generator_transfer assess "$run/retained.json.gz" \
  --data "$phoneme" --cache "$feature_cache" --speech "$weights/silero_vad.jit" \
  --encoder "$encoder" --fit "$run/fit" --retained "$sine" --fresh "$additional" --stereo "$stereo" \
  --recent-sine "$recent_sine" --recent-sine-receipt "$recent_sine_receipt" \
  --recent-bbc "$recent_bbc" --recent-bbc-receipt "$recent_bbc_receipt" --confirmation "$direction_media" \
  --anchor "$panels/audio-edit-direction-held-2026-11-16.json.gz" "$panels/audio-edit-direction-fresh-2026-11-16.json.gz"
"$runner" -B -m scripts.audio_joint_generator_vc prepare \
  "$phoneme" "$old_phonemedf" "$aligned_cache" "$backgrounds" "$run/fit" "$run/media" --extension "$extension"
"$runner" -B -m scripts.audio_joint_generator_vc score \
  "$run/media" "$run/fit" "$weights" "$encoder" "$run/conversion.json.gz"
```
