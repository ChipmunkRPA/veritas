# Native clocks in bounded PartialSpoof source ranges

The first 128 MiB of each publisher archive yielded six short edited recordings
and two matched ordinary controls across the train and development splits.
This intake verifies the original waveform clocks. The matched sample is too
small for a supervised fit or independent performance estimate.

## Source and selection

The PartialSpoof v1.2 publisher identifies the release as CC BY 4.0. Its
README credits the ASVspoof2019 database under ODC Attribution 1.0. The
older University of Edinburgh VCTK record underlying the ASVspoof data
also declares ODC Attribution 1.0. These are recorded separately because
ODC Attribution covers database rights and does not establish all individual
content rights. The PartialSpoof trial and annotation files lack the
donor and generator identities for the reconstructed edits. No audio,
model, or trained parameters accompany this publication.
[PartialSpoof release and terms](https://zenodo.org/records/5766198)
· [PartialSpoof construction](https://arxiv.org/abs/2204.05177)
· [ASVspoof2019 release](https://datashare.ed.ac.uk/handle/10283/3336)
· [VCTK original record](https://datashare.ed.ac.uk/items/3ea66d32-4541-4cf4-9571-443cc042466b/full)
· [ODC Attribution terms](https://opendatacommons.org/licenses/by/1-0/)

The intake keeps the earlier source audit's one-edit criterion. A positive has
one 0.50–1.20 second interval inside a 4–9 second recording, with at least
0.40 seconds of labeled genuine speech on either side. Ordinary controls
have the same duration bounds and no spoof-labeled intervals. Candidate
and control PCM must be complete native 16 kHz mono PCM16. Every labeled
boundary must be integral in samples, contiguous from zero, and inside the
waveform. The last label can end at most 15 samples before native PCM ends.
Those residual samples stay unlabelled. A control shares the declared
speaker, differs in duration by at most one second, and can be used once.
The closest unused control is selected in a fixed digest order, with
ties decided by digest. No detector response enters the selection.
[Metadata audit and labels](audio-partialspoof-source-audit-2026-11-13.md)

## Results

| Publisher split | Complete WAVs | Native short edits | Native eligible controls | Matched positive/control cases |
| --- | ---: | ---: | ---: | ---: |
| Train | 1,700 | 2 | 35 | 0 |
| Development | 1,633 | 4 | 38 | 2 |

All six eligible annotated edits and all 73 eligible genuine controls passed
the native clock check. The matched development cases each came from a
different declared speaker. The six eligible edited files have no declared
speaker overlap across splits. The pairs are different utterances by the
same declared speaker. There are no unedited parents for the exact edited
utterances, no noise simulation, and no known competing speech condition.

Increasing the prefix from 16 MiB per split, where no short edit qualified,
to 128 MiB per split yielded only two matched cases. A larger sequential
prefix could produce additional development samples, but the source still
lacks generator ancestry and exact parent/donor links. No detector ran.
[Native receipts and exact source positions](evaluation-panels/audio-partialspoof-native-intake-2026-11-13.json)

## Replay

Obtain the small publisher metadata files using the previous
[source audit instructions](audio-partialspoof-source-audit-2026-11-13.md#replay).
The native replay reads sixteen consecutive 16 MiB ranges, eight per split.
It verifies each range response, the source audit's first 64 KiB commitments,
the exact audio bytes, and the reported labels. The full 2 GB archives are
unnecessary. Keep the range cache local.

```sh
set -e
native=$(mktemp -d)
base=https://zenodo.org/records/5766198/files
chunk=16777216
for split in train dev; do
  for offset in 0 16777216 33554432 50331648 67108864 83886080 100663296 117440512; do
    stop=$((offset + chunk - 1))
    curl -fsSL --range "$offset-$stop" --max-filesize "$chunk" \
      -D "$native/$split-$offset.headers" \
      "$base/database_$split.tar.gz?download=1&chipmunk-range=$offset" \
      -o "$native/$split-$offset.range"
  done
done
PYTHONPATH=backend python3 -B -S -m scripts.review_audio_partialspoof_native "$metadata" "$native" \
  --compare docs/evaluation-panels/audio-partialspoof-native-intake-2026-11-13.json
```
