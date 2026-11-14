# Testing short speech edits against resynthesis and louder voices

This comparison uses the public SINE archive to measure a known source of
false alarms in short voice review. Each selected original is paired with
one alternate delivery. The alternate is either a resynthesis of the same
words, a synthetic replacement spliced into the original recording, or an
infilling edit. Every pair is examined clean and with a matched recording
of competing voices. [SINE dataset](https://huggingface.co/datasets/PeacefulData/SINE) ·
[Publisher's methods](https://arxiv.org/html/2501.03805#S3.SS2)

## Sources and controls

- Six 192 MiB prefixes at publisher revision
  `81be5d5e6a7d5d0f9bbbb613abed7b700f43d793` provide 36 matched pairs
  across 26 declared readers. The run takes at most one pair per reader
  per treatment, excludes the preview's single reader, and uses each
  utterance once. A reader may occur in more than one treatment. The
  receipt pins the complete prefix hashes, the publisher's object sizes
  and ETags, and each source WAV and PCM hash.
- The 12 cut-and-paste cases have a marked region lasting 0.28–0.77 s.
  Each six-second crop is selected by hash from the starts that retain
  the entire region on the publisher clock. The label is in centiseconds
  and maps to exact samples on the evaluation grid at that precision.
  These prefixes lack the infill label file. Infill receives a flag
  count only on the first six seconds of each edited delivery.
- Each pair uses the same six-second background excerpt at nominal
  −5 dB relative to the original, with one common peak-safe scale.
  The eight distinct rooms share no participants with each other or with
  the rooms named in earlier project evaluation documents inspected at
  `2fdcae31`. The room-only controls carry 2.02–3.58 seconds of overlapping
  annotated words. The room waveforms are real and the mixes are authored.
  [AMI terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

The publisher declares Apache 2.0 for the SINE release. Its original
recordings come from LibriVox via LibriLight. The authors' terminology
treats resynthesized speech as a control for changed spoken content even
though it has undergone synthesis processing. The paper identifies Vocos
for this processing while the dataset card says HiFi-GAN. The replay uses
the labeled published files and makes no claim to distinguish vocoders.
One editor lineage supplies both manipulated variants. The data establishes
neither generalization to other generators nor natural field accuracy.
No native audio is included in this repository.
[SINE license and labels](https://huggingface.co/datasets/PeacefulData/SINE#license) ·
[LibriVox public-domain terms](https://librivox.org/pages/public-domain/)

SPADE's publisher requires account approval and lists English as pending.
MIST's publisher identifies its release as research-only. SpeechSplice
declares CC BY 4.0 downstream but draws on ASVspoof 2019, whose underlying
terms need a separate review for this use. These sources supplied no
audio to this run.
[SPADE](https://huggingface.co/datasets/rogertseng/spade) ·
[MIST](https://huggingface.co/datasets/tung2308/MIST_SpeechInpaintingDataset) ·
[SpeechSplice](https://zenodo.org/records/14223054) ·
[ASVspoof source](https://datashare.ed.ac.uk/handle/10283/3336)

## Comparison

The 152 deliveries comprise 144 original and alternate views and eight
isolated backgrounds. The current short speech review and the two
interval-trained heads use the exact frozen points from the earlier
experiment. Each route sees only the delivered clip. Local flags must
cover at least half the marked edit and reach 0.30 temporal IoU to count
as useful. Because the current windows are two seconds long, edits shorter
than 0.60 seconds cannot meet its IoU requirement at any position.
[Earlier calibration](audio-interval-learning-2026-11-13.md) ·
[Fitted coefficients](evaluation-panels/audio-interval-learning-fit-2026-11-13.json)

| Outcome | Current clean / noisy | Clock head clean / noisy | Paired head clean / noisy |
| --- | ---: | ---: | ---: |
| Unedited originals flagged, out of 36 per condition | 0 / 1 | 0 / 4 | 0 / 4 |
| Same words resynthesized, flagged out of 12 per condition | 8 / 1 | 4 / 2 | 4 / 2 |
| Synthetic splice flagged, out of 12 per condition | 2 / 1 | 2 / 2 | 3 / 2 |
| Synthetic splice located usefully, out of 12 per condition | 0 / 0 | 1 / 1 | 1 / 1 |
| Infill delivery flagged, out of 12 per condition | 9 / 3 | 2 / 2 | 2 / 2 |

The noisy results repeat the same foregrounds from the clean condition.
Every route assessed all 144 foreground deliveries. The current check
left four of eight room-only controls unassessed. Each experimental
check left two unassessed. Every assessed room-only clip was unflagged.

The current check's clearest response was to processing: all 12 clean
resynthesized recordings scored above their untouched matches and
eight crossed the existing flag point. Competing speech reduced that
count to one. This is evidence that an elevated short voice response
can reflect synthesis processing of unchanged spoken content. It does
not tell us which part of a resynthesized recording drove the response.

Nine marked edits were shorter than 600 ms. Both experimental heads
selected **1.400–2.200 s** for one known **1.510–1.850 s** edit in both
conditions. The current check did not flag its clean view and flagged
the wrong interval in its noisy view. The other 11 edits yielded no
useful experimental location in either condition. Both heads also lost
seven clean infill flags and added three noisy-original flags. The
short range is worth testing on other source and editor families. These
results leave the serving detector and overall calibration as before.
[Full scalar results](evaluation-panels/audio-sine-resynthesis-transfer-2026-11-14.json)

## Replay

Run the scalar verifier using only standard-library Python. It checks
the paired denominators, fixed points, native interval geometry and
counts. The native path below additionally reopens the pinned source
bytes and reconstructs the clean, mixed and room-only inputs. Each
publisher archive request is bounded to 192 MiB. The AMI request takes
only eight headers and eight six-second PCM ranges.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
receipt=docs/evaluation-panels/audio-sine-resynthesis-transfer-2026-11-14.json
python3 -B -S -m scripts.audio_sine_resynthesis_transfer review "$receipt"

cache=$(mktemp -d)
rev=81be5d5e6a7d5d0f9bbbb613abed7b700f43d793
base=https://huggingface.co/datasets/PeacefulData/SINE
curl -fLsS --max-filesize 200000 "https://huggingface.co/api/datasets/PeacefulData/SINE/tree/$rev/data" -o "$cache/list.json"
for shard in 2 3 13 17 25 29; do
  curl -fLsS -H 'Accept-Encoding: identity' --range 0-201326591 --max-filesize 201326592 \
    -D "$cache/split-$shard-192.headers" \
    "$base/resolve/$rev/data/split-$shard.tar.gz?download=true&part=0-201326591" \
    -o "$cache/split-$shard-192.range" || exit 1
done
backend/.venv/bin/python -B -m scripts.audio_sine_resynthesis_transfer freeze "$cache/run" --cache "$cache"
backend/.venv/bin/python -B -m scripts.audio_sine_resynthesis_transfer score "$cache/run"
backend/.venv/bin/python -B -m scripts.audio_sine_resynthesis_transfer audit "$cache/run" --cache "$cache"
```

Local verification reconstructed all 152 delivery hashes and independently
reselected all 36 SINE pairs and eight room ranges from the source archive
and manual annotations. Native replay uses the pinned AMI annotations at the path supplied by
the [AMI overlap source runner](../backend/scripts/audio_locate_competing_speech.py)
and the previously installed local speech assets. The public receipt
contains source commitments, timestamps and scalar responses.
