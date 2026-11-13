# Matched speech under overlapping voices with two new synthesis families

This study tests whether examining the same noisy speech through two cleaned
views improves AI speech detection while keeping genuine flags low. The prior
ODSS panel used VITS and FastPitch–HiFiGAN. This transfer uses pre-existing
MeloTTS and Chatterbox TTS samples from PhonemeDF. [Earlier exact-prompt ODSS study](audio-odss-matched-room-2026-11-13.md)
· [PhonemeDF paper](https://aclanthology.org/2026.lrec-1.469/)

## Source and analysis

The publisher's CC BY 4.0 release lists 28,539 prompts with real speech from
LibriSpeech train-clean-100 and corresponding synthetic renditions. Fetching
only the first 96 MiB of the genuine, MeloTTS and Chatterbox archives yielded
311 complete matching utterance IDs across four source readers. All 28,539
transcript members were present in the genuine prefix. Compressed range sizes
and SHA-256 values are recorded for 36 accepted responses. These commitments
cover the observed ranges. The publisher's whole-archive MD5 values identify
the archive versions and require complete downloads to verify. [Publisher release and attribution](https://zenodo.org/records/19522979)
· [Publisher file metadata and license](https://zenodo.org/api/records/19522979)
· [LibriSpeech source license](https://www.openslr.org/12/)
· [VCTK source license](https://datashare.ed.ac.uk/items/30e7453c-9ea8-48b4-8e18-f96d0dc62928/full)
· [MeloTTS weights terms](https://huggingface.co/myshell-ai/MeloTTS-English)
· [Chatterbox weights terms](https://huggingface.co/ResembleAI/chatterbox)

We selected three prompts per source reader using a fixed hash ordering,
requiring the same release utterance ID, unique normalized transcripts, and
6–14 seconds per rendition with a longest-to-shortest ratio at most 1.25.
The delivered source is each original recording's first six seconds. The
model scores seconds 1.5–4.5 on each rendition's own clock. The complete
utterance is matched by publisher ID and text. The scored excerpts can contain
different words because delivery rates vary. The four groups represent
LibriSpeech readers and prompt clusters. A local comparison found zero shared
source reader IDs, normalized prompts, or genuine PCM hashes with either
ODSS role. The synthetic voice identities and any overlap with detector
training were unverified.

Each case places the target in front using BBC R&D's measured binaural room
response. An AMI competing-speech excerpt is placed left or right at
room-mean target-to-masker ratios of 0 and −5 dB. The matching three source
types receive the identical background, position and gain for each case.
Background-only and room-only recordings serve as controls. Twelve maskers
are mutually nonoverlapping and exclude the earlier ODSS excerpts from the
same AMI meeting. Source readers, prompt corpus and TTS families transfer to
the new panel. The physical room and competing-speech meeting are shared
with ODSS. [BBC static measurement rights and attribution](audio-bbc-room-assets-v303-method.md)
· [AMI terms and attribution](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

The study compares native mono, weighted prediction error dereverberation
(WPE), and a fixed frontal mid-side canceller. The same voice checkpoint and
speech admission logic score each view. Each individual rule uses its frozen
ODSS development point with zero genuine reader groups flagged on that
development data. The selective rule retains each native mono flag and adds
a flag only if the native score exceeds the secondary ODSS point and both
cleaned views each exceed their zero-reader development points.
All source and rendered window commitments and decision rules were fixed
before PhonemeDF model inference. [Commercial voice asset review](commercial-model-audit.md)

## Results

| Frozen noisy decision | MeloTTS detected | Chatterbox detected | Genuine flags | Genuine reader groups flagged | Background-only flags |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native mono | 7/24 | 14/24 | 7/24 | 4/4 | 2/24 |
| WPE | 19/24 | 19/24 | 14/24 | 4/4 | 0/24 |
| Frontal mid-side | 23/24 | 23/24 | 18/24 | 4/4 | 2/24 |
| Selective cleanup agreement | 17/24 | 19/24 | 10/24 | 4/4 | 2/24 |

The selective rule retained the 21 mono synthetic flags and added 15,
including 10 MeloTTS and five Chatterbox cases. It added three genuine flags
and zero background-only flags. At −5 dB it detected 15/24 synthetic mixtures
and flagged 4/12 genuine mixtures, compared with 8/24 and 3/12 for mono. At
0 dB it detected 21/24 synthetic mixtures and flagged 6/12 genuine mixtures,
compared with 13/24 and 4/12 for mono. The two mono background flags came
from the same masker at different levels. WPE admitted 22 of 24 background
controls. Every foreground analysis was admitted across all views.
Room-only mono flagged 13/24 synthetic controls and 0/12 genuine controls.

The native score ranked Chatterbox above its matched genuine recording in
20/24 noisy pairs and MeloTTS higher in 15/24. The frontal view ranked
20/24 and 17/24 respectively. WPE ranked 19/24 and 18/24. The gain in
flagged synthetic views reflects both score shifts and modest ranking changes.

Both cleanups improved separation from the known target waveform. Frontal
cancellation improved all 72 noisy target mixtures, with median SI-SDR gains
of 2.02 dB at nominal 0 dB and 2.37 dB at −5 dB. WPE improved 65 of 72,
with median gains of 0.91 and 1.09 dB. The alternate FFT reproduced all 96
frontal noisy views exactly. WPE differed by at most one signed 16-bit unit.

The selective rule gained substantially more synthetic flags than genuine
flags in this transfer. Retrospective ODSS yielded five additional synthetic
flags and five additional genuine flags, with genuine flags rising from 2/16
to 7/16. The fresh noisy native result already flagged genuine samples from
all four reader groups, so the same group count offers little reassurance.
These repeated renders come from 12 prompts, four readers, one room and one
competing-speech meeting. The cleaned views are correlated because they use
the same mixture and classifier. Retain the selective rule for further
calibration and leave the customer score unchanged. [Anonymous scalar receipt](evaluation-panels/audio-phonemedf-matched-transfer-2026-11-13.json)

## Reproduce

Store the publisher data outside the repository. The anonymous receipt contains
commitments and scalar responses. Raw recordings, source IDs and transcripts
are kept out of the receipt. A reader using only the Python standard library
recomputes frozen development points, source and condition denominators, and
paired outcome counts. The model runner rebuilds the aligned waveform
commitments from the local publisher files.

Download 12 consecutive 8 MiB ranges starting at byte zero from each
publisher file, and retain response headers next to their data. Use names
`real/0.range`, `real/0.headers`, `MeloTTS/0.range`, and
`ChatterboxTTS/0.range`, with offsets in bytes. The runner requires each
range's full recorded byte count and exact `Content-Range`. For example, the
first genuine range uses the URL below. Use the captured release API response
for byte-identical replay, since live download counts can change. [Captured public publisher response](evaluation-panels/audio-phonemedf-release-19522979.json)

```sh
curl -fSs --max-filesize 8388608 -H 'Accept-Encoding: identity' \
  --range 0-8388607 -D "$cache/real/0.headers" \
  'https://zenodo.org/api/records/19522979/files/PhonemeDF_real.tar.gz/content?chipmunk-offset=0&chunk=8388608' \
  -o "$cache/real/0.range"
```

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VERITAS_OFFLINE=true
for phase in freeze score replay; do
  backend/.venv/bin/python -B -m scripts.audio_phonemedf_matched_transfer "$phase" \
    --cache "$cache" --record "$release_record" --run "$run" \
    --surrey "$surrey_room" --bbc "$bbc_room" --ami "$ami_sources" || exit 1
done
python3 -B -S backend/scripts/review_audio_phonemedf_matched_transfer.py \
  "$receipt" docs/evaluation-panels/audio-odss-matched-room-2026-11-13.json
```
