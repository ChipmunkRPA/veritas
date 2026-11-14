# Background downweighting for short soundtrack timing leads

A short timing edit can duplicate a piece of speech within a single soundtrack.
The existing experimental witness looks for matching audio near the edit and
compares the mouth images at both occurrences. This study tests whether
downweighting persistent background frequencies helps when a short repeat is
mixed with rain, train noise, or a competing speaker. A repeated sound paired
with different mouth pictures is a lead for inspection. Authorship, direction
of change, the full edit boundary, and phoneme-level agreement are separate
questions. [Previous repeat comparison](video-repeated-speech-bands-2026-11-14.md)
· [Noise-sensitive frequency weighting](https://www.isca-archive.org/interspeech_2021/salvati21_interspeech.pdf)
· [Lip-focused content consistency](https://arxiv.org/abs/2610.30872)

## Method

The comparison ran the existing two-band repeat witness, a fixed three-band
lag contrast, and the same lag contrast after one clip-wide frequency filter.
The filter reduces power in frequency bins with persistent energy, using the
30th percentile of power from 32 ms frames at 16 ms spacing. It applies a single
bounded spectral weight across the whole excerpt so that both copies of the
same speech undergo identical processing. It uses only the submitted acoustic
signal. The new lag contrast takes the median correlation across 180–950,
950–3,500, and 3,500–5,900 Hz. It requires a core response of at least 0.53
and a lead of at least 0.08 over alternative lags at least 24 ms away. All arms
share the same speech activity and mouth movement rules. The existing face
geometry weights have a pinned MIT license. [Weight license](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

The source plan retains four prior comparison excerpts and selects two
different 2.52-second sections per held Wikitongues recording. The five held
recordings were previously used at other offsets. Each excerpt produces two
staged shifts of 200 or 280 ms and both PCM and actual 48 kbps AAC deliveries.
Controls include untouched recordings, gain, filtering, noise, constant delay,
a repeated background, and keeping the picture synchronized with the local
audio shift. The latter controls are independently retimed, encoded, decoded,
and measured. Rain, train and competing speech are mixed into the original,
the two shifts and the two synchronized controls at about 5 dB above voice
RMS. The original sound, pictures, audio range commitments, and asset licenses
are bound in the numeric receipt. The repository contains the measurements and
hashes. [Sound sources and terms](audio-focused-window-transfer-2026-11-12.md#frozen-comparison)

Both reported repeats need temporal intersection over union of at least 0.50
with their known ranges. To advance as a supplementary check, a variant
needed three added noisy AAC locations from at least two recordings, at most
one extra control flag per carrier, and zero added flags on synchronized or
constant-shift controls. The comparison between the two new arms isolates the
effect of persistent-frequency downweighting.

## Held results

These counts apply separately to PCM and AAC. The deliveries reuse the same
recordings and edit plans across both carriers.

| Condition | Prior useful | Fixed contrast useful | Downweighted useful | Controls flagged, prior / fixed / downweighted |
| --- | ---: | ---: | ---: | ---: |
| Clean | 4/20 | 1/20 | 1/20 | 1 / 2 / 2 of 90 |
| Competing speech | 0/20 | 0/20 | 0/20 | 0 / 0 / 0 of 30 |
| Rain | 0/20 | 0/20 | 0/20 | 0 / 0 / 0 of 30 |
| Train | 0/20 | 0/20 | 0/20 | 0 / 0 / 0 of 30 |

One of ten planned held excerpts lacked a complete shared clock and two more
lacked sufficient mouth observations. They remain in the denominator. Each arm
assessed 14 of 20 shifts per condition, plus 63 of 90 clean controls and 21 of
30 controls per noisy condition. Each new arm retained one of the four clean
locations in each carrier and added no useful recoveries. Each also flagged
two previously unflagged controls per carrier. Both candidates stay offline.
[Matched results](evaluation-panels/video-whitened-sound-transfer-2026-11-14.json)

The local app and Standard PDF now show which parts of the decoded soundtrack
have no timing measurement, including gaps beyond the scheduled windows. The
JSON's existing sample intervals and source clock can reproduce the displayed
ranges. [Coverage and source replay](video-timing-window-coverage-2026-11-14.md)

## Replay

The first command independently recounts all paired outcomes without media
dependencies. For native replay, supply the original comparison and held
recordings and the pinned `rain.ogg` and `train.ogg` files. The source URLs,
hashes and attributions are in the receipt and its parent source records.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 -B -S -m scripts.video_whitened_sound_transfer review \
  docs/evaluation-panels/video-whitened-sound-transfer-2026-11-14.json
python3 -B -S -m scripts.video_whitened_sound_transfer freeze "$run/plan.json"
backend/.venv/bin/python -B -m scripts.video_whitened_sound_transfer run "$run/receipt.json" \
  --freeze "$run/plan.json" --comparison "$comparison" --held "$held" --sounds "$sounds"
```
