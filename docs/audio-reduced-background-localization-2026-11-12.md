# Locate short speech edits after reducing background sound

The local three-second background review gives a response for one complete
window. This experiment asks whether adjacent responses can narrow the
review to a shorter part of a longer recording. It preserves the existing
voice decisions, tests the contribution of a denser window schedule
separately, and counts both located edits and genuine time flagged.
[Existing three-second check](audio-nemotron-noise-transfer-2026-11-12.md) ·
[Temporal localization study and boundary limitations](https://arxiv.org/abs/2609.10051) ·
[Research implementation of the temporal metrics](https://github.com/sami42200/tdlmc-audio-deepfake-localization/blob/main/src/tdlmc/metrics.py) ·
[Speech enhancement and detector error](https://arxiv.org/abs/2603.14767)

## Frozen comparison

Use 36 distinct publisher source-speaker groups with exact native clocks
from the CC BY 4.0 LlamaPartialSpoof release. Eighteen files contain partial
edits and six are fully synthetic, with three partial files and one full
file from each of six generator families. Twelve files are genuine.
The first archive ranges held too few previously unused edited groups for
the intended completely source-separated test. This transfer uses 15
previously unused groups from the two earlier comparison rosters, including
all 12 genuine groups and three partial-edit groups. It also uses 21
previously exposed groups. Two partial source files recurred, and one exact
gain-only delivery recurred. All 72 added mixtures use different recorded
backgrounds. Training exposure remains unknown. This roster can measure
method transfer and does not qualify a change for serving.
[Publisher data and license](https://zenodo.org/records/14214149) ·
[Earlier partial speech comparison](audio-partial-publisher-transfer-2026-11-12.md)

Each file contributes a gain-only control and two mixtures at −5 dB
source-to-added-background RMS with a common peak-safe gain. The scenes
are a construction recording and a market recording that may contain
intelligible conversation. Four eight-second excerpts from each scene
also enter alone. A separate implementation reconstructed all 116 input
waveforms exactly. Its worst measured difference from the target level was
0.000011 dB after rounding up.
[Construction by Dsw4, public domain](https://commons.wikimedia.org/wiki/File:Construction.ogg) ·
[Leicester Market by NotFromUtrecht, CC BY-SA 3.0](https://commons.wikimedia.org/wiki/File:Leicester_Market_binaural_recording_(May_2011).ogg)

The baseline checks its existing two or three nonoverlapping three-second
windows. The proposed schedules check three-second windows every 1.5
seconds. A tail outside the final grid window remains unassessed by this
schedule. The native-only arm marks the intersection of two adjacent
AI-leaning responses. The processed arm also accepts an elevated response
on a background-reduced copy of an inconclusive or speech-unassessed
window, using the current background review point and the same
speech admission. An original-leaning window cannot be rescued. A fourth
arm preserves every current range and adds the processed-arm ranges.
These are dependent views scored by the same voice classifier. Their
intersection locates the evidence for inspection.

A useful partial location covers at least half the same labeled edit and
reaches 0.30 intersection over union with it. A fully synthetic file
counts when any location is flagged. Count genuine flagged samples on
ordinary files and on the genuine portions of partial files, taking the
union of reported ranges first. Also retain file, source-group, and
generator counts. The frozen gate asks for at least four added noisy
partial groups from at least three generators across both scenes,
at most one added noisy genuine group, zero added clean genuine or
isolated flags, and at most two extra percentage points of genuine
sample time marked in the noisy deliveries.

## Local results

| Delivery | Useful partial locations, current / denser native / denser processed / combined | Genuine flags, current / denser native / denser processed / combined |
| --- | ---: | ---: |
| Gain-only | 7 / 14 / 14 / 14 of 18 | 0 / 0 / 0 / 0 of 12 |
| Construction, −5 dB | 0 / 2 / 10 / 10 of 18 | 0 / 0 / 4 / 4 of 12 |
| Market, −5 dB | 1 / 3 / 9 / 9 of 18 | 1 / 3 / 6 / 6 of 12 |

The combined check found useful new noisy locations across **11 of 18
partial source groups**, including all three groups outside both prior
panels. The gains span both scenes and all six generator families. It added
flags on **six distinct genuine sources** and marked an additional **48.2
seconds of genuine time**, or **9.8 percentage points** of the 492.4
genuine seconds in the noisy test. That duration includes genuine parts
of partially edited recordings. No isolated background received a range.
Four market-only excerpts passed processed speech admission without a flag.
The method exceeded both false-alarm limits and stays outside the app.

Denser native sampling alone gained seven useful locations on the gain-only
views, preserved all seven incumbent useful locations, and produced zero
genuine file flags there. Reducing background sound added no useful
location on those views. This supports a separate confirmation of the
native-only schedule on new sources and ordinary clean, quiet-noise and
recompressed controls. The gain-only signal is a matched laboratory
control. The comparison does not establish that customer recordings can
be routed reliably by the presence of background sound.

The completed run checked **727 original windows** and processed **304**
eligible windows. The portable reader recounts the fixed decisions and
flagged sample time without the model or recordings. Its native replay
also checks each exact original and processed window hash.
[Evaluation receipt](evaluation-panels/audio-reduced-background-localization-2026-11-12.json)

```sh
python3 -B -IS backend/scripts/review_audio_reduced_background_localization.py review \
  docs/evaluation-panels/audio-reduced-background-localization-2026-11-12.json
```

## Reproduce the frozen inputs

Use the pinned 192 MiB prefix from the earlier study and fetch the exact
additional range and slices below. These ranges contain uncompressed tar
bytes despite the publisher filename. The reader starts a mid-archive slice
at a verified 512-byte WAV member header. Verify the bytes against the
receipt before comparing results. Keep the original speech, authored mixes,
and processed windows local.

```sh
publisher=https://zenodo.org/records/14214149/files
curl -fLsS --range 201326592-369098751 --max-filesize 167772160 \
  "$publisher/R01TTS.0.a.tgz?download=1&cohort=temporal-speech-extension" -o "$sources/part-a-192m-352m.tarseg"
for pair in '2 2147483648-2248146943' '6 6442450944-6543114239' '10 10737418240-10838081535'; do
  number=${pair%% *}
  bytes=${pair#* }
  curl -fLsS --range "$bytes" --max-filesize 100663296 \
    "$publisher/R01TTS.0.a.tgz?download=1&cohort=slice-${number}g" -o "$sources/slice-${number}g.tarseg" || exit 1
done
curl -fLsS --max-filesize 8000000 \
  'https://upload.wikimedia.org/wikipedia/commons/2/2c/Construction.ogg' -o "$sources/construction.ogg"
curl -fLsS --max-filesize 3500000 \
  'https://upload.wikimedia.org/wikipedia/commons/d/d9/Leicester_Market_binaural_recording_%28May_2011%29.ogg' -o "$sources/leicester.ogg"
```

```sh
export PYTHONPATH=backend VERITAS_BILLING_MODE=disabled VERITAS_AUDIO_DETECTOR_MODE=candidate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_reduced_background_localization prepare "$run" \
  --archive "$sources/part-a-192m.prefix.tar" --extension "$sources/part-a-192m-352m.tarseg" \
  --labels "$labels" --backgrounds "$sources" --slice "$sources/slice-2g.tarseg" \
  --slice "$sources/slice-6g.tarseg" --slice "$sources/slice-10g.tarseg"
backend/.venv/bin/python -B -m scripts.audio_reduced_background_localization primary "$run"
backend/.venv/bin/python -B -m scripts.audio_reduced_background_localization enhance "$run" --model "$model" --overlay "$overlay"
backend/.venv/bin/python -B -m scripts.audio_reduced_background_localization processed "$run"
python3 -B -IS backend/scripts/review_audio_reduced_background_localization.py export "$run" --out "$receipt"
```
