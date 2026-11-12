# Speech detection on public RADAR files and matched loud backgrounds

## Source and fixed comparison

The RADAR 2026 evaluation release provides authentic and synthetic speech
labels after assorted media transformations. The evaluation metadata
identifies language, duration and labeled intervals. Individual noise,
codec, generator and speaker IDs are unavailable in those tables. The
publisher released the data under CC BY 4.0. This study uses a bounded
32 MiB prefix of the official evaluation archive and the matching official
label and language files. It is a transfer check on previously unused
project inputs. The archive prefix is a convenience sample, and possible
model-training overlap or repeated speakers remain unknown.
[Publisher release and terms](https://zenodo.org/records/22110815) ·
[Challenge methods](https://arxiv.org/abs/2605.09568)

Select English files with a single label spanning the entire recording.
Within each label and official duration range, choose the first 16 files by
salted hash order. Short files span 2.5–4 seconds and longer files span
6–14 seconds. The source roster, exact FLAC hashes and method were sealed
before any new voice scores were computed. All 64 files have distinct
decoded PCM hashes, continuous 16 kHz clocks, and decoded durations within
three samples of the publisher values.

Use the existing Full-profile speech adapter with its fixed local
calibration. The existing adapter may score windows on a longer file but
supplies whole-file confidence only within its qualified short scope. The
candidate applies the existing three-second assessment separately to the
first and last non-overlapping three-second intervals. Files at least nine
seconds long also use the central three seconds. The candidate records
local corroboration if at least two intervals lean AI and none lean
original. It yields no overall probability or original-file tendency.
The follow-up criterion is at least eight corroborated synthetic files and
at most two genuine flags out of 16 in each class. The thresholds and
weights stay fixed throughout the experiment.
[Installed voice assets and terms](commercial-model-audit.md)

For each short file, make two matched versions with newly acquired ordinary
recordings. Use street traffic recorded by ElemePodcast under CC BY 4.0 and
café ambience recorded by Marble Toast under CC0 1.0. Decode each original
to mono 16 kHz. The frozen offsets assign four excerpts per scene evenly
across each label. Add each selected background at **−5 dB source-to-added-
background RMS ratio** before quantization. The RADAR files may already
contain sound and processing, so the ratio describes the full original
file against the added background. Apply one common peak-safe gain to the
unoverlaid control and both mixtures, then quantize to signed 16-bit
samples. The matched gain-only controls account for the level change.
The test uses two recorded scenes with repeated excerpts. It cannot
assign an effect to the undisclosed original processing.
[Traffic source and credit](https://commons.wikimedia.org/wiki/File:Audio_sound_of_a_busy_traffic_in_Porthacourt.ogg) ·
[Café source and credit](https://commons.wikimedia.org/wiki/File:Cafe_ambiance.ogg)

This paired evaluation follows recent work that measures foreground speech
and environmental audio separately when assessing mixed recordings. It
uses no new external detector weights.
[MixFake study](https://arxiv.org/abs/2605.23201) ·
[Modality-aware audio benchmark](https://arxiv.org/abs/2608.09593)

## Initial results

Each short row contains 16 files per source class. The AI flag counts for
genuine speech are false alarms. Inconclusive and unassessed responses
remain visible.

| Short delivery | Synthetic AI / original / inconclusive / unassessed | Genuine AI / original / inconclusive / unassessed |
| --- | ---: | ---: |
| Original RADAR bytes | 11 / 1 / 3 / 1 | 1 / 8 / 2 / 5 |
| Matched gain-only control | 11 / 1 / 3 / 1 | 0 / 8 / 3 / 5 |
| Added traffic | 3 / 0 / 5 / 8 | 1 / 0 / 3 / 12 |
| Added café ambience | 0 / 0 / 5 / 11 | 0 / 0 / 2 / 14 |

The long-file adapter was inconclusive on all 16 synthetic files and 15
of 16 genuine files. One genuine file was unassessed. The fixed candidate
found corroborating AI-leaning evidence in **10/16 synthetic** and
**0/16 genuine** longer files. At least two portions were assessed in
13 files of each class. Allowing just one AI window would have found
14/16 synthetic and 0/16 genuine in this initial sample. The
two-window policy stayed fixed for the separate confirmation.
[Initial score and source ledger](evaluation-panels/audio-radar-transfer-2026-11-12.json)

The independent verifier reopened the 32 MiB prefix and found 408 complete
FLACs. It reconstructed the 64 exact source selections and clocks, 73
native three-second views, 64 matched mixtures and 32 matched gain-only
controls. It reproduced the counts from the saved responses without
running the model. It rejected a changed window hash and a changed
advancement result.

## Separate longer-file confirmation

The separate publisher development split derives from a different source
collection. We selected 20 bona fide and 20 spoof English clips from
a new 32 MiB prefix using the same 6–14 second duration range and a
frozen hash order. Every selected original is native mono 16 kHz 16-bit
FLAC. Independent inspection verified 40 complete source clocks, 98
planned non-overlapping portions, and no exact PCM overlap with the
initial panel. Speaker and generator identities remain unavailable.
No classifier point or measurement changed between trials.

| Longer-file confirmation | Synthetic (20) | Genuine (20) |
| --- | ---: | ---: |
| Current whole-file inconclusive | 20 | 20 |
| At least two portions assessed | 18 | 20 |
| At least one AI-leaning portion | 18 | 0 |
| Corroboration from two or more portions | 18 | 0 |

The frozen confirmation required at least ten synthetic files with
corroboration and allowed up to two genuine false flags. It passed. This
qualifies a narrowly scoped local evidence check. It does not measure a
whole-file probability, the boundary of a partially synthetic section,
or performance after the matched loud backgrounds in the short panel.
[Confirmation score and source ledger](evaluation-panels/audio-radar-long-confirmation-2026-11-12.json)

## Local evidence

The Full audio worker runs the supplemental check on continuous native
mono 16 kHz 16-bit FLAC or PCM WAV lasting 6–14 seconds. It reports the assessed
three-second ranges on the original sample clock and supplies a playable
clip and marked waveform for each corroborating AI-leaning window. The
overall audio conclusion is separate. A result with no corroboration shows
the window readings without publishing clips as corroborating evidence.

Three isolated Full analyses used simulated prepaid checkout. The two
synthetic files showed respectively two and three corroborating windows
with exact audio and waveform evidence. The genuine file showed its
three local window readings without a corroborating flag or clip. All
three whole-file conclusions stayed inconclusive. The API, JSON, Standard
PDF and signed source replay agreed on the original sample positions.
Chrome played all five three-second excerpts, decoded their waveform
images and confirmed that changing the active excerpt pauses the previous
one. The signed packages also verified with model and network imports
disabled.

The WAV extension reopens the 72 pinned longer FLACs from both publisher
splits as native 16-bit WAV copies. Both containers produced the same
decoded PCM, sample clocks and 171 previously scored window input hashes.
Two isolated Full analyses with simulated prepaid checkout reproduced the
earlier FLAC results. The synthetic WAV had three corroborating ranges
with matching playback and marked waveforms. The genuine WAV had three
window responses and no corroborating evidence. Both whole-file results
were inconclusive. The API, JSON, Standard PDF and original-file signed
replay agreed. This verifies lossless container transfer and carries zero
overall confidence weight. [WAV parity and app receipt](evaluation-panels/audio-long-wav-parity-2026-11-12.json)

To repeat the PCM and window comparison, use the same scratch source
directory described below after reconstructing both panels.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.review_audio_long_wav_parity "$source_dir"
```

## Reproduce the source selection

Download only the first 32 MiB of the public archive. The runner requires
the exact prefix and metadata hashes. Downloads and derived audio stay in
the local working directory.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VERITAS_AUDIO_DETECTOR_MODE=candidate
source_dir=$(mktemp -d)
radar=https://zenodo.org/records/22110815/files
curl -fLsS --range 0-33554431 --max-filesize 33554432 \
  "$radar/RADAR2026-eval.tar.gz?download=1" -o "$source_dir/archive-prefix.gz"
curl -fLsS "$radar/label_RADAR2026-eval.txt?download=1" -o "$source_dir/labels.txt"
curl -fLsS "$radar/lang_RADAR2026-eval.txt?download=1" -o "$source_dir/languages.txt"
curl -fLsS "$radar/LICENSE.txt?download=1" -o "$source_dir/LICENSE.txt"
curl -fLsS 'https://upload.wikimedia.org/wikipedia/commons/2/2d/Audio_sound_of_a_busy_traffic_in_Porthacourt.ogg' \
  -o "$source_dir/traffic.ogg"
curl -fLsS 'https://upload.wikimedia.org/wikipedia/commons/5/54/Cafe_ambiance.ogg' \
  -o "$source_dir/cafe.ogg"
run_dir="$source_dir/run"
backend/.venv/bin/python -B -m scripts.audio_radar_transfer freeze "$run_dir" --source "$source_dir"
backend/.venv/bin/python -B -m scripts.audio_radar_transfer score "$run_dir"
backend/.venv/bin/python -B -m scripts.audio_radar_transfer review "$run_dir/result.json"
python3 -I -S backend/scripts/review_audio_radar_transfer.py "$source_dir" "$run_dir"
```

For the separate confirmation, acquire the development archive prefix and
metadata in the same working directory.

```sh
curl -fLsS --range 0-33554431 --max-filesize 33554432 \
  "$radar/RADAR2026-dev.tar.gz?download=1" -o "$source_dir/dev-prefix.gz"
curl -fLsS "$radar/label_RADAR2026-dev.txt?download=1" -o "$source_dir/dev-labels.txt"
curl -fLsS "$radar/lang_RADAR2026-dev.txt?download=1" -o "$source_dir/dev-languages.txt"
confirmation="$source_dir/confirmation"
backend/.venv/bin/python -B -m scripts.audio_radar_long_confirmation freeze \
  "$confirmation" "$run_dir/result.json" --source "$source_dir"
backend/.venv/bin/python -B -m scripts.audio_radar_long_confirmation score \
  "$confirmation" "$run_dir/result.json"
backend/.venv/bin/python -B -m scripts.audio_radar_long_confirmation review \
  "$confirmation/result.json" "$run_dir/result.json"
python3 -I -S backend/scripts/review_audio_radar_transfer.py "$source_dir" "$confirmation" \
  --confirmation-prior "$run_dir/result.json"
```
