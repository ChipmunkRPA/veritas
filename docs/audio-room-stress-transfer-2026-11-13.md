# Separate room acoustics from competing speech in partial voice detection

The earlier Aachen trial recovered cleaner waveforms from overlapping speech without recovering useful voice detections. This test applies room acoustics and competing speech separately to the same six-second clips. It measures whether the prior blind stereo separator adds useful timestamps as background speech increases. The classifier, speech gate, and analysis windows stay fixed. [Aachen transfer](audio-spatial-mask-aachen-transfer-2026-11-13.md) · [Controlled re-recording study](https://www.isca-archive.org/interspeech_2025/muller25_interspeech.pdf) · [Noise-aware evaluation](https://arxiv.org/abs/2512.13744)

## Matched sound conditions

The seven prior Aachen speech edits span six synthesis families. Each retains the same crop, room, competing excerpt, and publisher-labeled intervals. Eight genuine readers and six background-only excerpts serve as assessment controls. These groups appeared in the earlier Aachen comparison. [Publisher speech and labels, CC BY 4.0](https://doi.org/10.5281/zenodo.14214149) · [Aachen impulse responses, MIT](https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/) · [AMI competing speech, CC BY 4.0](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml) · [Commercial model audit](commercial-model-audit.md)

The first three conditions are dry speech, an early room response ending about 8 ms after the earliest main arrival with a 1 ms taper, and the full measured response. Match foreground root mean square across these variants. The remaining conditions add the same competing excerpt at foreground-to-background ratios of +5, 0, −5, and −10 dB in the stereo mean. One gain applies to all seven variants in each group. Labels stay on the authored clock. The rendered direct path arrives 6–9 ms later. An input audit reproduced all 167 waveform hashes. Across 92 noisy renders, the worst difference from the target SNR was 0.00006 dB. The worst foreground level difference was also below 0.00006 dB. No samples clipped.

## Operating points and timing

Use the rounded stereo mean for all conditions and both anonymous outputs of the prior blind separator for noisy and background-only conditions. The two analyzed windows are 1–3 and 3–5 seconds. A useful result selects the strongest admitted window above the route's fixed point, covers at least half a labeled edit, and reaches 0.30 interval intersection over union. The primary comparison reuses prior mono and separator points. Before assessment inference, freeze a second separator point at the greatest response across four SNRs on eight genuine development readers. Four were newly selected and all eight are disjoint from assessment readers. Both roles use the same two rooms and AMI sessions, with distinct excerpts.

The separator must add two edits from two families at each of two adjacent SNRs to advance. Across all SNRs it may add at most one genuine reader group and zero background-only groups.

## Results

Each cell gives correctly localized edits and flagged ordinary groups. The seven speech edits and eight genuine controls repeat across all seven speech conditions. The separate background-only row has six genuine excerpts.

| Paired condition | Mono | Prior blind separator | Mono plus prior separator | Mono plus newly fitted separator |
| --- | ---: | ---: | ---: | ---: |
| Dry | 5 / 0 | — | — | — |
| Early room response | 5 / 0 | — | — | — |
| Full room response | 3 / 1 | — | — | — |
| +5 dB | 1 / 1 | 2 / 0 | 3 / 1 | 1 / 1 |
| 0 dB | 2 / 2 | 1 / 1 | 3 / 3 | 2 / 2 |
| −5 dB | 0 / 1 | 1 / 1 | 1 / 2 | 0 / 1 |
| −10 dB | 0 / 0 | 1 / 0 | 1 / 0 | 0 / 0 |
| Background only | — / 2 | — / 0 | — / 2 | — / 2 |

Every case had an admitted mono window. Dry and early-room audio each also produced one above-point response at the wrong location. No other mono or separator flag selected a wrong edit window.

The early response preserved all five dry locations, with a median score change of −0.025 log-odds across the seven labeled targets. Extending to the full room response lost two and changed the median by −3.320. Adding competing voices changed which edit was found as well as how many were found. At −5 and −10 dB, mono found no correct locations. These are paired observations in two simulated acoustic conditions, with seven repeated edits from one publisher.

At the earlier separator point, adding the separator recovered three unique edit groups across the four noisy levels. The same 3.000–5.000 s window covered the 3.267–4.292 s insert at both −5 and −10 dB. A second source had a labeled 3.173–4.073 s insert in that window at +5 and 0 dB, and a third had a 2.760–4.885 s insert at +5 dB. The separator added one genuine reader at 0 dB and another at −5 dB. It added no background-only flags. Two gains occurred at +5 dB only, short of the required two adjacent levels.

The separately fitted separator point was 7.267491 versus 6.760728 earlier. It yielded zero added locations and zero added ordinary flags. Both candidates failed the advancement gate. This supports evaluating reverberation-robust scoring before adding another separator. A prospective test would keep matched genuine and partial-edit versions together, render each across new rooms, and measure ordinary-edit flags and timestamp overlap under the same paired ladder.

## Scalar replay

The receipt records anonymous source groups, exact label intervals, admitted model responses, and the separately frozen development point. The standard-library reader verifies the frozen points and independently recomputes the tables, paired changes, and advancement rule from 106 development and 424 assessment forwards. Reproducing model inference additionally requires the native source roster and local voice assets. [Scalar receipt](evaluation-panels/audio-room-stress-transfer-2026-11-13.json)

```sh
python3 -B -IS backend/scripts/review_audio_room_stress_transfer.py \
  docs/evaluation-panels/audio-room-stress-transfer-2026-11-13.json
```
