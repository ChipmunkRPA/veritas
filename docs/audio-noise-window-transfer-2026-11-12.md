# Staggered speech windows with spectral noise reduction

The comparison tests two changes separately. The first shifts three-second
windows by 750 milliseconds and retains the 1.5 seconds shared by three
adjacent AI-leaning responses. The second measures a noise-reduced copy of
eligible windows on the same original clock. It uses the installed voice
classifier. It makes no new calibrated probability or inference about the
background's origin.
[Boundary and segment research](https://arxiv.org/html/2604.02913v1) ·
[Out-of-domain localization research](https://arxiv.org/html/2507.03468v1) ·
[Installed checkpoint terms](commercial-model-audit.md)

## Fixed source and comparison rules

Part B of the CC BY 4.0 LlamaPartialSpoof release supplies 18 distinct
partial-edit speaker groups with three recordings from each of six
generators. Twelve use cut and paste and six use overlap and add. All 18
publisher speaker codes occur in earlier project comparisons. The exact 18
publisher member IDs, WAV hashes and native PCM hashes do not recur in the
three linked comparisons. Model-training exposure remains unknown. The
study scanned 845 complete Part B members and excluded one member with an
invalid publisher label clock. It retained all admitted labels on the exact
16 kHz source clock. Twenty-four Part A genuine groups form separate
selection and held controls. None of those speakers occurs in the 18
held edited groups. The genuine sources come from prior project archives.
[Publisher license and label files](https://zenodo.org/records/14214149) ·
[Prior source ledgers](audio-reduced-background-localization-2026-11-12.md)

Each source contributes a gain-matched reference and two noise mixtures.
The foreground is 5 dB quieter than the added background by RMS before
quantization. All three views use one peak-safe gain. The separate
selection recordings are restaurant ambience and a coastal street. The
held recordings are Vienna traffic and Cebu market. These backgrounds
have prior project use. Two excerpts of each sound enter alone in its
assigned role. Excerpts can overlap. Counts represent two recordings
per role, rather than independent acoustic environments.
[Restaurant by stephan, public domain](https://commons.wikimedia.org/w/index.php?title=File:Restaurant_ambience.ogg&oldid=1003321006) ·
[Coastal street by Cori Samuel, public domain](https://commons.wikimedia.org/w/index.php?title=File:Sunday_in_the_city_street_noise1.ogg&oldid=1110023161) ·
[Vienna by Matthias A. Bertsch, CC BY-SA 4.0](https://commons.wikimedia.org/w/index.php?title=File:2023-Mai-20_14h34min_Wien_innerer_Lerchenfelder_G%C3%BCrtel_Thaliagasse_Nachmittagsverkehr.wav&oldid=982248618) ·
[Cebu market by Bim24, CC BY-SA 4.0](https://commons.wikimedia.org/w/index.php?title=File:Carbon_Market_at_Night_in_Cebu_City.ogg&oldid=1169036785)

The original comparators are the current sparse long-window rule and the
earlier 1.5-second grid. The comparison is offline and excludes other
current report checks. No recording has exactly six seconds. The staggered
grid uses the same original calibration and reports a range only when all
three windows support it. The reduced arm retains all original AI-like
windows and can add a response after inconclusive or unassessed windows
that have no underlying original tendency. Require at least half the
reduced window to contain detected speech. The cutoff is the largest
admitted reduced response from the selection controls, determined before
scoring any held recording.

Estimate each frequency bin's noise from its 20th percentile power across
the submitted recording. Use a 512-sample Hann analysis window with a
128-sample hop. Subtract 1.5 times the noise estimate in power, retain
at least 35% of the original amplitude and smooth the gain over nearby
bins and frames. Retain the original complex phase and exact sample count.
This implementation uses no trained enhancement weights or upstream code.
Changing noise or competing speech can remain in the output.
[Spectral subtraction method](https://doi.org/10.1109/TASSP.1979.1163209)

Evaluate individual ranges at 50% coverage of a labeled edit and 0.30
temporal intersection over union. Count all highlighted genuine samples
after deduplicating overlaps, including genuine parts of edited recordings.
Unassessed files stay in the denominator. The fixed comparison asks for
four added noisy edit groups across three families, up to two additional
genuine groups under noise, no added clean or isolated group and up to
three additional percentage points of genuine time highlighted.

## Held results

| Delivery | Useful edit locations, sparse / dense / staggered / reduced / combined | Genuine files with a flag, same order |
| --- | --- | --- |
| Gain-only reference | 4 / 11 / 13 / 13 / 13 of 18 | 0 / 0 / 0 / 0 / 0 of 12 |
| Cebu market, −5 dB | 2 / 3 / 2 / 2 / 3 of 18 | 2 / 3 / 3 / 3 / 4 of 12 |
| Vienna traffic, −5 dB | 2 / 3 / 1 / 1 / 3 of 18 | 0 / 0 / 0 / 0 / 0 of 12 |

No arm flagged any of the four isolated background excerpts. The 94 held
deliveries produced 998 original window responses, including unassessed
windows. Of 411 eligible reduced responses, zero exceeded the fixed point
from 66 selection windows. The 74 staggered triplets came entirely from
the original waveforms. Reducing background sound added zero locations.

Under loud noise, the dense original grid localized six of 36 edits, up
from four on the sparse grid. It highlighted 17.61 of 502.36 genuine seconds,
down from 25.91, and flagged one additional genuine market delivery. The
combined grid kept the same six useful locations as dense alone. It added
one genuine market group flag and highlighted 4.15 more genuine seconds,
an increase of 0.83 percentage points. It met the false-alarm and genuine
time limits but missed both detection-gain requirements. Both new arms
remain experimental and the overall confidence is unchanged.

The gain-only reference offers the clearest signal for further native
window work. The staggered grid located 13 of 18 edits with no additional
genuine file flags. It also marked 6.09 more genuine seconds inside the
edited files than the dense grid. Retest both grids on new speakers and
separately recorded low-noise backgrounds. The gain-only inputs do not
establish a dependable way to route customer recordings by noise level.
[Held receipt](evaluation-panels/audio-noise-window-transfer-2026-11-12.json)

## Replay

The public receipt stores decisions and score commitments without media.
The offline reader reconstructs the calibrated original direction, rescue
veto, time ranges and reported totals. With the retained local run it checks
every exact input PCM hash and all stage commitments. It cannot reproduce
model inference or confirm the speech-activity decision on reduced audio.
The separate dependency receipt records 14 matching helper files from the
pre-existing application revision. The local reader verified all 134 input
PCM hashes and the stage and dependency commitments.
[Dependency receipt](evaluation-panels/audio-noise-window-transfer-dependencies-2026-11-12.json)

```sh
python3 -B -I -S backend/scripts/review_audio_noise_window_transfer.py \
  docs/evaluation-panels/audio-noise-window-transfer-2026-11-12.json \
  --dependencies docs/evaluation-panels/audio-noise-window-transfer-dependencies-2026-11-12.json
python3 -B -I -S backend/scripts/review_audio_noise_window_transfer.py \
  "$run/receipt.json" --run "$run" --dependencies "$run/dependency-review.json"
```

To rerun the model, fetch the three exact Part B byte ranges with their
published checksums in the runner. Verify HTTP 206 and the range total
`12791859200`. Use the Part A ranges from the
[previous native speech study](audio-reduced-background-localization-2026-11-12.md#reproduce-the-frozen-inputs).
Retain the reviewed sound recordings under the names in the runner.
Use the same local voice and speech-activity weights.

```sh
publisher=https://zenodo.org/records/14214149/files
curl -fLsS --max-filesize 16000000 "$publisher/label_R01TTS.0.b.txt?download=1" -o "$sources/part-b.labels.txt"
curl -fLsS --range 0-33554431 --max-filesize 33554432 "$publisher/R01TTS.0.b.tgz?download=1&study=staggered-windows-32m" -o "$sources/part-b-32m.prefix"
curl -fLsS --range 33554432-100663295 --max-filesize 67108864 "$publisher/R01TTS.0.b.tgz?download=1&study=staggered-windows-extension" -o "$sources/part-b-extension"
curl -fLsS --range 2147483648-2214592511 --max-filesize 67108864 "$publisher/R01TTS.0.b.tgz?download=1&study=staggered-windows-slice" -o "$sources/part-b-slice"
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export VERITAS_AUDIO_DETECTOR_MODE=candidate VERITAS_BILLING_MODE=disabled
backend/.venv/bin/python -B -m scripts.audio_noise_window_transfer prepare "$run" \
  --sources "$sources" --part-a "$part_a" --labels "$part_a_labels" --sounds "$sounds"
backend/.venv/bin/python -B -m scripts.audio_noise_window_transfer run "$run"
```
