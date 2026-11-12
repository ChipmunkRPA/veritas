# Local voice coverage under recorded street noise

The source-only comparison found 8 of 144 intervals with strong speech-band masking. It flagged two intervals that lacked a measurable foreground. Both trials missed the frozen coverage and false-warning criteria. The observation remains experimental. [Both completed aggregate receipts](evaluation-panels/audio-background-limited-coverage-2026-11-11.json)

## Frozen comparison

The earlier enhancement test recovered 44 synthetic flags and introduced 39 false AI flags. This comparison leaves the voice input unchanged and tests a separate coverage observation. It measures whether a short interval resembles the nearby sound outside detected speech and whether that observation tracks strong interference measured from known foreground and background components. The observation carries zero authorship-confidence weight. [Enhancement results](audio-traffic-enhancement-transfer-2026-11-11.md) · [Noise-spectrum estimation research](https://numana.uni-koeln.de/sites/kunoth/user_upload/Mathematical_Foundations_Data_Analysis/SS19/Materials/IMCRA.pdf)

Eight LibriSpeech test-clean readers and six prepared synthetic voices form 14 foreground groups. The synthetic roster has two voice groups each from clone, expressive, and stock cohorts. The known source groups and exact clean PCM are distinct from the preceding traffic, two-background, and enhancement comparisons. The prepared synthetic samples had previous project exposure elsewhere. Each 3-second test contains 2.30 seconds of foreground centered between two 350 ms reference spans. A fixed 25 ms ramp softens the boundaries. Add contiguous street recordings at nominal −6 and +6 dB center SNR. The initial trial used full-band RMS. After it produced few masked speech-band intervals in two scenes, a second freeze used the same exact source slices and set the target in the 250 Hz to 4 kHz speech band. The candidate and decision points remained unchanged. Both trials include one clean view per foreground and two isolated excerpts per background. Each has 104 exact views, including 84 matched mixtures. The two trials share their foreground and background groups. [LibriSpeech and CC BY 4.0](https://www.openslr.org/12/) · [Synthetic source preparation and rights](audio-short-edit-source-plan-v169.md)

Finn Årup Nielsen’s Lyngby street recording and the Gransee street recording published by SchlurcherBot are new to this comparison series and both licensed CC BY-SA 4.0. Gransee includes prominent birdsong and distant vehicles. A new segment of Cori Samuel’s public-domain street recording supplies the third background and has no time overlap with the earlier enhancement excerpts. The publisher SHA-1, file length, terms, and decoded sample digests are bound. Raw recordings, derived mixtures, and per-clip measurements stay local. [Lyngby source and terms](https://commons.wikimedia.org/wiki/File:Lyngby_Hovedgade,_2015-01-21.ogg) · [Gransee source and terms](https://commons.wikimedia.org/wiki/File:Ambient_sound_Mauerstrasse_Gransee_birds_and_traffic_2026-05-22.oga) · [Sunday source and terms](https://commons.wikimedia.org/wiki/File:Sunday_in_the_city_street_noise1.ogg)

The candidate uses the current speech activity checkpoint on the original waveform. It requires at least 224 ms of detected speech in the center, at most 80 ms in either reference, and a reference above −54 dBFS in both flanks. It also requires reference spectra to agree within 5 dB in at least three of four frequency bands between 250 Hz and 4 kHz. The centered speech region is partitioned into four half-open 575 ms intervals. The candidate compares median band powers of each interval to the flanks. It flags an interval when its overall band power is no more than 2.5 dB above the averaged reference and three bands stay within 4 dB after the overall shift. It records the exact original and reference sample positions and reuses the current voice activity and screening path. [Frozen runner](../backend/scripts/audio_background_limited_coverage.py) · [Voice and activity checkpoint terms](commercial-model-audit.md)

The separate control labels use the isolated components. A bin needs foreground energy of at least the larger of the fixed floor and 1.6% of that foreground’s maximum bin energy. Among these bins, speech-band SNR at or below −3 dB is the masked control. At or above +3 dB is the audible control. Intermediate and low-foreground bins retain their own labels. These are exact acoustic energy comparisons on controlled mixtures. No human intelligibility judgments were collected. Advancement requires at least 40% masked-bin recall in two scenes, matched bins in at least four genuine and four synthetic groups, at most 10% false warnings on audible bins and at most 5% on absent bins.

## Local results

| Measurement | Full-band mixture levels | Speech-band mixture levels |
| --- | ---: | ---: |
| Masked genuine intervals flagged / planned | 5/21 | 4/76 |
| Masked synthetic intervals flagged / planned | 7/17 | 4/68 |
| Audible intervals falsely warned / assessed / planned | 0/6/306 | 0/10/180 |
| Absent-foreground intervals falsely warned / assessed / planned | 3/3/38 | 2/2/38 |
| Recordings with an assessed reference / planned | 8/104 | 7/104 |
| Scenes meeting masked-interval recall requirement | 0/3 | 0/3 |

The initial trial found all 12 correct intervals in Lyngby, which supplied 37 of its 38 masked intervals. The speech-band trial supplied 50 masked intervals in Lyngby, 49 in Gransee and 45 in Sunday. The eight warnings still all came from Lyngby. Only 13 of 144 known masked intervals had an assessed reference under the candidate. In the stronger −6 dB speech-band mixtures, 33 of 42 recordings lacked the required speech anchor. The false warnings came from low-foreground intervals within mixed recordings. None of the six isolated background recordings qualified for a warning.

At −6 dB in the speech-band trial, the incumbent left 23 of 24 genuine and 17 of 18 synthetic inputs unassessed. Both admitted inputs were inconclusive. The proposed cue identified four of these unassessed recordings, two genuine and two synthetic. It did not overlap any existing directional result in either trial, including two genuine false AI calls in the initial trial and one in the speech-band trial. The cue stays out of the service.

The initial and speech-band runs used 107 and 57 voice classifier forwards respectively, each with 9,776 activity forwards. The second scoring stage took 46.48 seconds on CPU. A separate SciPy periodogram implementation checked 416 interval measurements per trial and rebuilt all 84 mixtures per trial sample-for-sample. It checked 12 and eight true masked-interval commitments. The private demonstrations include the exact submitted waveform with reference and review spans, the exact selected WAV samples, and a JSON map on the 16 kHz source clock. [Independent review command](../backend/scripts/review_audio_background_limited_coverage.py)

## Local replay

Use the earlier traffic, two-background and enhancement private rosters and the LibriSpeech test-clean archive. Save the Wikimedia originals and corresponding `imageinfo` JSON replies with `sha1`, `url`, `size`, and `extmetadata` in a new directory. The runner expects `lyngby.ogg`, `gransee.oga`, `sunday.ogg`, and JSON files named `lyngby.json`, `gransee.json`, and `sunday.json`. The initial run's earlier code and receipts were retained locally. The current runner supports both mixture scales.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_background_limited_coverage freeze "$run_dir" \
  --mix-band speech_band --librispeech "$test_clean_archive" --prior-traffic "$traffic_roster" \
  --prior-background "$background_roster" --prior-enhancement "$enhancement_roster"
backend/.venv/bin/python -B -m scripts.audio_background_limited_coverage run "$run_dir"
backend/.venv/bin/python -B -m scripts.audio_background_limited_coverage readout "$run_dir"
```

The independent reviewer accepts either completed trial. It uses NumPy and SciPy to reconstruct mixtures, recompute band powers, check stored voice calibration decisions, and verify interval and group counts without running model inference. It reads `freeze.json`, `roster.json`, `measurements.json`, `aggregate.json`, `private-evidence.json`, and the local `pcm` and `noise` directories. Use `--published` to check the exact retained receipt against the published aggregate.

```sh
backend/.venv/bin/python -I -B backend/scripts/review_audio_background_limited_coverage.py "$run_dir" \
  --published docs/evaluation-panels/audio-background-limited-coverage-2026-11-11.json
```
