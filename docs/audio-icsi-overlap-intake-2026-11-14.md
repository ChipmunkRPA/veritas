# ICSI competing-speech source bank

This local bank contains **24 six-second excerpts** from three different ICSI
meetings. The dialogue-act annotations identify 16 participants across the
three meetings with no shared declared identifiers. The bank has 12 excerpts
with at least two simultaneous annotated speakers for 1.23–1.99 seconds and
12 comparison excerpts without annotated overlap. All contain at least 4.5
seconds of annotated speech. The source recordings are original meetings.
[ICSI dataset and CC BY 4.0 terms](https://groups.inf.ed.ac.uk/ami/icsi/download/)

The runner enumerates disjoint six-second ranges using the official core
word tiers. It joins word tracks to participant IDs through the published
dialogue acts. Meetings with unbound timed tracks are excluded from the
participant-separated selection. Fixed hash order chooses the meetings and
samples before accessing the audio. The archive replays with SHA-256
`cf4860245b9ca9c9ed11a66e5a74cfea2a99a686a52f7580aa0a6bc2225895a9`.

Acquisition reads each WAV header and the exact selected byte ranges from
one pinned mirror revision. All files are continuous native 16 kHz mono
16-bit PCM. Each excerpt has 96,000 samples, source position, annotation
overlap ranges and a decoded PCM commitment. The public intake receipt
contains source identifiers, timings and hashes. A second HTTP client
re-fetched and matched all 2,304,000 native samples. The audio and words
stay in scratch. [Pinned audio mirror](https://huggingface.co/datasets/aaronyee/icsi-meeting-corpus/tree/ea358f7db63cd68c59b22d7990918f326480215f)
· [Source receipt](evaluation-panels/audio-icsi-overlap-intake-2026-11-14.json)

## Replay the source bank

Download the official version 1.0 core file as `core.zip`. Store the
recordings and artifacts in a separate private directory. The second command
makes bounded public range requests. The last command verifies the labels,
clock extents and saved PCM offline with standard-library Python.

```sh
mkdir -p "$scratch"
curl -fLsS 'https://groups.inf.ed.ac.uk/ami/ICSICorpusAnnotations/ICSI_core_NXT.zip' \
  -o "$scratch/core.zip"
export PYTHONPATH=backend
python3 -B -S -m scripts.audio_icsi_overlap_intake plan "$scratch" "$scratch/core.zip"
python3 -B -S -m scripts.audio_icsi_overlap_intake acquire "$scratch" "$scratch/core.zip"
python3 -B -S -m scripts.audio_icsi_overlap_intake replay "$scratch" "$scratch/core.zip"
cmp "$scratch/receipt.json" docs/evaluation-panels/audio-icsi-overlap-intake-2026-11-14.json
```
