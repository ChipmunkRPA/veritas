# Overlapping speech windows on paired neural edits

This transfer pilot tests the denser original-speech schedule on a second
publisher corpus. The current three-second windows remain the comparator.
The denser schedule advances by 1.5 seconds. A separate combined arm retains
every current interval and adds the overlap of two adjacent AI-leaning
windows. All measurements use the same pinned voice calibration. The
comparison changes neither the calibration nor the app.
[Earlier window comparison](audio-reduced-background-localization-2026-11-12.md) ·
[Current model and terms](commercial-model-audit.md)

## Paired source files

PartialEdit V1.1 supplies edited recordings and exact segment labels from
VoiceCraft and SSR-Speech. Only the two published subsets enter this test.
Use every matched utterance in both pinned archive prefixes that meets the
publisher evaluation-speaker split, length and known-edit rules. The two
edited versions share an utterance identifier. Fetch the corresponding
microphone 1 recording from the official Edinburgh VCTK 0.92 release as a
genuine control. The datasets are CC BY 4.0. The original version uses its
own recorded duration. The comparison uses publisher edit labels on each
edited version's own clock. It makes no claim that the original and edited
file share the same resampling or sample alignment.
[PartialEdit V1.1 and license](https://zenodo.org/records/18829689) ·
[PartialEdit methodology](https://www.isca-archive.org/interspeech_2025/zhang25g_interspeech.html) ·
[VCTK 0.92 and license](https://doi.org/10.7488/ds/2645)

The prefixes yielded 18 matched utterances across three anonymous VCTK
speaker groups. All three groups occur in the project's earlier audio
history. None of the exact 18 utterance keys appears in the two checked
historical ledgers, the local audio v9 source plan and the public v4–v6
development receipt. Earlier model-training exposure is unknown. These
are 18 matched utterances from three source groups, even when each file
appears in several delivery conditions. The source coverage falls short of
the eight groups needed even to consider confirmation.
[Public historical receipt](evaluation-panels/audio-v4-v6-development.json)

Each exact source appears as canonical mono PCM16 at 16 kHz. The original
is decoded once and the released edited float WAV is quantized once.
The 54 clean recordings each have a counterpart with construction sound at
−5 dB and with market sound at +5 dB source-to-background RMS. The
mixtures can receive a peak-safe gain. The clean recordings keep their
decoded gain. The scenes use fixed offsets, repeat earlier backgrounds,
and differ in both sound and mixing level. They cannot isolate an effect
of level or background alone. No customer recording enters this test.
[Construction by Dsw4, public domain](https://commons.wikimedia.org/wiki/File:Construction.ogg) ·
[Leicester Market by NotFromUtrecht, CC BY-SA 3.0](https://commons.wikimedia.org/wiki/File:Leicester_Market_binaural_recording_(May_2011).ogg)

Usefulness is measured on the individual reported intervals. A useful
interval covers at least half of one labeled edit and has at least 0.30
temporal intersection over union with it. False time counts the union of
flagged intervals on original recordings and on the genuine parts of
edited recordings. An unassessed window yields no flag and stays in
the denominator. The sparse schedule uses the same native sampling and
corroboration rule as the app. This offline comparison does not test
production dispatch, latency or report coverage. No file landed at exactly
six seconds, so the separate two-second review contributes no results.

## Local results

| Delivery | Useful locations, current / denser / combined | Genuine recordings flagged, current / denser / combined | Genuine seconds flagged, current / denser / combined |
| --- | ---: | ---: | ---: |
| Clean | 5 / 17 / 19 of 36 | 0 / 0 / 0 of 18 | 101.28 / 70.38 / 125.44 |
| Construction, −5 dB | 0 / 1 / 1 of 36 | 0 / 0 / 0 of 18 | 0 / 0.90 / 0.90 |
| Market, +5 dB | 3 / 8 / 9 of 36 | 0 / 0 / 0 of 18 | 25.70 / 16.76 / 38.88 |

Each scene has 383.81 seconds of genuine material, including genuine time
inside edited recordings. In this pilot every flag on genuine time came from
a partially edited file. On clean audio the denser schedule alone located
12 more edited deliveries and flagged 30.90 fewer genuine seconds than
the sparse schedule. The combined schedule preserved all current
locations and added 14 useful deliveries, while adding 24.16 seconds of
genuine time. That is 6.29 percentage points of genuine clean time.
At −5 dB only one of 36 known edits was localized. For market mixtures
the denser-only arm localized five more deliveries and marked 8.94 fewer
genuine seconds. The clean and market gains span both editing methods.

The completed run measured 771 native windows across 162 deliveries.
At least one window was assessed on 161 deliveries. The other delivery
was a construction mixture. The denser-only result supports further
source-separated comparison with processing and transport controls.
Strong-background misses and the three previously exposed speaker groups
leave it outside the app. [Frozen outcomes](evaluation-panels/audio-partialedit-dense-transfer-2026-11-12.json)

## Replay

The runner accepts exact compressed publisher ranges. Verify the response
is HTTP 206 and matches the requested range before saving it. SHA-256 pins
for every input are in the runner and receipt. The official VCTK archive is
almost 12 GB. Only its central directory and the 18 needed member ranges
are used. The runner validates the exact range, ETag, ZIP local header,
member size and CRC before decoding. Keep publisher audio and intermediate
files local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export VERITAS_AUDIO_DETECTOR_MODE=candidate
publisher=https://zenodo.org/records/18829689/files
vctk=https://datashare.ed.ac.uk/server/api/core/bitstreams/535f4286-e54c-4038-838c-a02285e32cb2/content
curl -fLsS "$publisher/PartialEdit_E1E2.csv?download=1" -o "$sources/labels.csv"
curl -fLsS "$publisher/PartialEdit_spk.zip?download=1" -o "$sources/speakers.zip"
curl -fLsS --range 11734391133-11747302976 --max-filesize 12911844 \
  "$vctk?veritas-range=11734391133" -o "$sources/vctk-directory"
for family in E1 E2; do
  curl -fLsS --range 0-83886079 --max-filesize 83886080 \
    "$publisher/$family.tar.gz?download=1&scope=prefix" -o "$sources/$family.prefix" || exit 1
done
curl -fLsS --range 83886080-251658239 --max-filesize 167772160 \
  "$publisher/E1.tar.gz?download=1&scope=extension" -o "$sources/E1.more"
curl -fLsS --range 83886080-335544319 --max-filesize 251658240 \
  "$publisher/E2.tar.gz?download=1&scope=extension" -o "$sources/E2.more"
for phase in acquire prepare score report; do
  backend/.venv/bin/python -B -m scripts.audio_partialedit_dense_transfer \
    "$phase" "$run" --sources "$sources" --backgrounds "$backgrounds" || exit 1
done
python3 -B -I -S backend/scripts/review_audio_partialedit_dense_transfer.py "$run/receipt.json"
```
