# Similar moving-picture sequences within one upload

Chipmunk Verify can show two time ranges where the same sequence appears to repeat despite lossy compression. The ranges, four original endpoint pictures, and supporting transitions appear in the video evidence view, PDF, and JSON. A signed evidence ZIP can replay the comparison against the supplied original. Loops, recurring footage, and ordinary edits can produce a match. This evidence carries zero weight in the AI estimate.

The existing exact sequence check runs first. The original similar-sequence profile covers the first 180 native frames of an unrotated 320 × 180 H.264 or VP9 video in 8-bit 4:2:0 format. It requires verified 50 ms spacing throughout the checked prefix. A qualifying exact result takes precedence, including one elsewhere in the wider exact scan. A follow-up trial qualified 640 × 360 at 25 fps. Other delivery profiles, shorter inputs, later ranges, and changes to playback speed or framing fall outside these tests. [360p transfer results](video-motion-sequence-360p-transfer-2026-11-12.md)

## What the comparison measures

The comparison averages each frame into a 40 × 30 luminance grid. Corresponding pictures in each pair must differ by at most 1.6 gray levels in root-mean-square distance. It also compares the signed spatial change between adjacent pictures, after removing the average change in brightness. Supporting transitions must pass the activity floor (0.4 gray levels), have a cosine similarity of at least 0.65, and have a smaller-to-larger activity ratio of at least 0.55.

A pair needs two disjoint ranges with at least half a second between the first and last timestamps, at least eight supporting transitions, support in at least half the transitions, and at most three transition positions between supports. The longest qualifying pair wins, with ties resolved by the earliest source positions. The JSON includes support offsets relative to each range's first frame, independent pixel commitments for both occurrences, and the original timestamps and images for all four endpoints. Prior research also compares spatial appearance and temporal information in video copy localization. This check uses project code and requires no external model weights. [Primary research](https://arxiv.org/abs/1811.10762)

## Local validation

The development panel used six earlier VCD2 camera recordings. The held panel used eight distinct retained camera originals from different camera types outside that roster. Source SHA-256 digests confirmed disjoint recordings. Both panels contain one lossy copy and five matched controls per source and codec. The controls are unchanged footage, a conventional cut to flipped footage, a still-frame hold, a reversed excerpt, and a repeated excerpt with altered frame spacing. Held excerpts span 19, 23, or 29 frames. A copy counts as useful when both reported intervals have at least 0.50 intersection over union against the authored intervals. These are paired observations from eight held sources, with possible overlap in participants. [Held capture roster](video-vcd2-unseen-controls-v287-result.md) · [VCD2 data terms](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA)

The 27 development combinations varied appearance distance, motion activity, and direction. The selected setting found 5 of 6 copies with zero flags on 30 controls in each codec. The held gate required at least four additional useful source groups over the exact baseline per codec, no loss of a useful exact match, at most one extra control flag, and zero new flags on holds or spacing changes. The settings were sealed before any held delivery was processed.

| Held comparison | H.264 copies | H.264 control flags | VP9 copies | VP9 control flags |
| --- | ---: | ---: | ---: | ---: |
| Exact pixels and timing | 0/8 | 0/40 | 0/8 | 0/40 |
| Similar pictures and timing | 7/8 | 11/40 | 7/8 | 11/40 |
| Similar pictures, timing, and motion | 7/8 | 0/40 | 8/8 | 0/40 |

The held encodes used FFmpeg 7.0.2 with H.264 CRF 23 and VP9 CRF 31. Independent FFmpeg frame hashing and a separate comparison recomputed all 17,280 native pictures and the outcomes on all 96 held deliveries. Local simulated checkout and analysis on the H.264 MP4 and VP9 WebM versions of held source 6 showed the two ranges at **1.150–2.050 s** and **6.300–7.200 s**. All eight original endpoints matched the PDF and JSON, and both signed packages replayed. Chrome displayed four original endpoint pictures in the plain evidence view and sought to 6.300 s from its jump link. A held-picture and a changed-spacing control showed no similar-sequence packet. [Case-level aggregate](evaluation-panels/video-lossy-sequence-transfer-2026-11-12.json)

The panel estimates behavior under these two encodes and control types. A single miss on H.264 remains. Ordinary repeated shots and loops can match the same criteria. The panel supplies no accuracy estimate for face swaps, AI authorship, or edits that leave no repeated sequence.

## Reproduce the comparison

The arithmetic replay uses only the public case rows. Independent native replay additionally needs the locally retained deliveries and probe ledgers. The evaluation code can regenerate those deliveries from the qualified VCD2 archive using the development stage before the held stage.

```sh
cd backend
.venv/bin/python -B -m scripts.review_video_lossy_sequence_transfer ../docs/evaluation-panels/video-lossy-sequence-transfer-2026-11-12.json
```
