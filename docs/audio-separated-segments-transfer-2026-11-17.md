# Short speech edits in a third measured stereo room

The [earlier stereo comparison](audio-separated-segments-2026-11-17.md)
located six of 64 labeled speech edits by combining two estimated views,
up from two with its angular estimate alone. The joint rule flagged five
of 32 ordinary mixtures, up from three. This transfer tests the same
strict settings on a different source corpus and measured stereo paths.

## Source clocks and ordinary controls

Fourteen previously verified same-utterance pairs supply seven disjoint
reader pairs. Their edited clips come from LlamaPartialSpoof and their
matched originals come from LibriTTS. Their speaker IDs are absent from
the earlier separator fit and transfer groups. The utterances appeared
in a separate project experiment that tested longer speech windows.
Six publisher-labeled generation families contribute original edits.
One labeled segment from each verified pair lasts 250–1,245 ms.
[LlamaPartialSpoof data and license](https://zenodo.org/records/14214149) ·
[LibriTTS source license](https://www.openslr.org/60/) ·
[Waveform verification](audio-six-ten-context-2026-11-16.md#same-utterance-controls)

Each six-second voice uses the matched original outside its labeled
interval. The synthetic version inserts the exact corresponding samples
from the publisher's edited audio. The ordinary edit runs the same
interval of the original through an 8 kHz round trip. Both replacements
use the same 20 ms fades. Each scene is delivered with neither version
edited, both ordinarily edited, either voice synthetically edited, or
both synthetically edited. The untouched inputs determine gain once.
The quieter voice is 10 dB below the competing voice in the stereo mean,
apart from rounding. Two placement schedules test partly overlapping
and staggered edit clocks.

The third room uses six pinned left and right measurements for
Huddersfield's three source positions. Convolution uses the paired
impulses at 16 kHz. Authored edit labels include the direct arrival.
The inference takes only the stereo mixture and does not read source
paths, angle labels or original tracks.
[Room release and license](https://zenodo.org/records/2639677) ·
[Pinned source files](audio-room-assets-v299-method.md)

## Frozen comparison

The same short-segment speech checkpoint reads the mono mean, up to three
anonymous angular estimates, and two independent estimates. The
independent separator retains the 2,048-sample unwhitened option. Each
response compares an interval containing at least half detected speech
with the higher score from its two adjacent 320-millisecond windows.
A usable location covers at least half the known edit and
has at least 0.30 intersection over union. Each edit counts once.
The comparison retains the prior strict points for each separate view.
It advances only if the pair of separated views adds at least two
locations across at least two speaker pairs, with at most two additional
ordinary alerts. This comparison measures timing. It does not attribute
a change to a particular speaker or change an overall assessment.
[Segment-aware detection method](https://arxiv.org/abs/2601.21925) ·
[Checkpoint and backbone terms](audio-frame-logit-localization-2026-11-17.md#separate-segment-trained-checkpoint)

## Held results

Each strategy returned an eligible score on all 70 six-second deliveries.
The 42 edited deliveries contain 56 true edit intervals. The 28 ordinary
deliveries comprise 14 untouched and 14 with two resampled intervals.

| View | Correct locations | Missed edits | Ordinary alerts | Edited deliveries with only a misplaced alert |
|---|---:|---:|---:|---:|
| Mono mean | 0/56 | 56/56 | 0/28 | 0/42 |
| Angular estimates | 0/56 | 56/56 | 0/28 | 0/42 |
| Independent estimates | 0/56 | 56/56 | 1/28 | 1/42 |
| Angular and independent estimates | 0/56 | 56/56 | 1/28 | 1/42 |

One independent estimate flagged 0.80–0.96 seconds in the same mix with
and without an edit. The edit was at 2.426–2.956 seconds. The separated
views added zero correct locations across zero reader pairs and one
ordinary alert. The measured gain from the earlier panel did not transfer.
This method stays in evaluation. The two source corpora and three measured
rooms provide controlled timing tests. These limited rooms and reader pairs
leave everyday stereo recordings and mono deliveries untested by the
separation comparison.
[Measured receipt](evaluation-panels/audio-separated-segments-transfer-2026-11-17.json.gz)

## Replay

The public receipt records every edit interval, source and rendered audio
fingerprint, the frozen points and selected local ranges. Raw speech and
model files remain local. The portable check reconstructs the
source freeze from the measured rows, verifies reader disjointness and
recounts the paired changes. A second native render produced byte-identical
inputs. A separate recount checked exact PCM sums for all 70 deliveries
and reproduced every timing and ordinary-alert total.

```sh
export PYTHONPATH=backend
panel=docs/evaluation-panels/audio-separated-segments-transfer-2026-11-17.json.gz
python3 -B -S -m scripts.audio_separated_segments_transfer review "$panel"
```

To rebuild the exact inputs, use the `matched-panel` from the same-utterance
preparation and `holdout` from the pinned room-asset preparation. The
receipts bind their source hashes. Native inference requires the same
verified local driver and inference-only checkpoint as the earlier stereo
comparison.

```sh
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
runner=backend/.venv/bin/python
module=scripts.audio_separated_segments_transfer
"$runner" -B -m "$module" prepare "$matched_panel" "$huddersfield" "$out/inputs"
"$runner" -B -m "$module" measure "$out/inputs" "$checkpoint" "$driver" "$out/held.json.gz"
```
