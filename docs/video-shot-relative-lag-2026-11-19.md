# Testing relative speech timing across a camera cut

The existing short speech and mouth comparison needs one continuous shot. Its separate near-silence check can inspect detected shots for dropouts. This trial asks whether the same original-frame mouth and speech representations can identify a signed change in alignment between two adjacent shots. Recent work separates audio offset from sound-specific articulation and measures correspondence close to the mouth. This experiment uses only the existing local assets. [PVSync](https://arxiv.org/abs/2610.09223) · [Frame-level speech and mouth consistency](https://arxiv.org/abs/2610.30872)

The frozen bank pairs twelve new GRID recordings from distinct retained speaker archives into six source groups. The recordings and exact pairings are separate from the pinned prior source rosters. The speakers have earlier project exposure. Each 4.8-second AVC MP4 has two 60-frame shots at native 25 fps, with the cut at frame 60 (2.400 s, PTS 216,000 on the 1/90,000 clock). Sound was encoded to mono 48 kHz AAC. Each group supplies a synchronous cut, an aligned longer pause made by repeating eight quiet starting pictures and the corresponding sound in the second shot, the same signed shift applied within both shots, and two files with the second-shot sound advanced or delayed by 200 or 280 ms. The first and second shots never share model input windows across the cut. [GRID publisher and licence](https://zenodo.org/records/3625687)

The comparison requires the two proposed offsets to differ by at least 160 ms. It reads each offset on exactly the same visible and audible rows in each half of each shot. The weakest of the four paired contrasts must exceed the frozen point. Competing directions with similar strength abstain. A correct estimate needs the exact observed cut and a signed lag change within 60 ms of the authored shift.

| Held condition | Files | Existing near-silence findings | Candidate findings | Correct signed change |
| --- | ---: | ---: | ---: | ---: |
| Second-shot sound advanced | 6 | 0 | 5 | 2 |
| Second-shot sound delayed | 6 | 0 | 5 | 2 |
| Synchronous cut | 6 | 0 | 4 | — |
| Longer aligned pause | 6 | 0 | 3 | — |
| Same lag within both shots | 6 | 0 | 3 | — |

All 30 inputs supplied an exact detected cut and enough common paired views. The existing near-silence route assessed ten inputs and recorded no interval. The candidate supplied four correct signed changes across four groups, six wrong signed estimates on edited files, seven alerts on synchronized cuts or pauses, and three alerts on equal-lag controls. Three edited mistakes gave the opposite sign. Three gave the expected sign but missed the 60 ms tolerance. One source pair selected the same 740 ms shift after a normal cut, after the aligned pause, and after both positive edit directions. This suggests that the strongest offset can follow the two recordings themselves. The frozen ordinary, equal-lag and wrong-result requirements failed, so the candidate stays offline. The uploaded research files represent staged timing changes, with classification of AI authorship outside this comparison.

Independent decoding verified all 3,600 original delivered picture timestamps and RGB digests, 2,304,000 exact presented audio samples and all 60 before, at and after mouth-crop anchors attached to the 20 selected findings. The local landmark runtime required macOS graphics access for the measured run. No media or individual source identities are included in the public scalar record.

The separate [29.97 fps transfer](video-shot-lag-point21-transfer-2026-11-19.md)
tests the stricter operating point and a different approach to measuring
each shot's timing independently.

```sh
python3 -I -S backend/scripts/review_shot_relative_lag.py \
  docs/evaluation-panels/video-shot-relative-lag-2026-11-19.json
```
