# Moving-picture repeat detection at 360p and 25 fps

This trial transfers the frozen appearance and motion rule from 320 × 180 at 20 fps to a native 640 × 360, 25 fps delivery. It compares the exact pixel baseline with the same fallback rule over the first 225 frames. The physical scan duration remains nine seconds. There is no model fitting and no external weights. [Previous motion-sequence validation](video-lossy-sequence-transfer-2026-11-12.md)

## Frozen comparison

The candidate retains the 40 × 30 luminance grid, the 1.6-gray-level picture-distance ceiling, the 0.4-gray-level motion floor, the 0.65 motion cosine and the 0.55 relative activity floor. The two intervals must have matching picture spacing, be disjoint, span at least half a second, and have at least eight supporting transitions accounting for at least half the range. The exact result takes precedence. A candidate needs verified 40 ms spacing on every adjacent frame in the prefix. At 25 fps the appearance descriptor averages 16 × 12 native pixels into each grid cell.

The panel uses six VCD2 camera originals from the retained v197 roster. Their digests differ from all 14 sources used in the earlier repeat study. They had prior project exposure in other video work. Each camera supplies a repeat, an unchanged control, a horizontal-flip cut, a held frame, a reversed excerpt, and a copied excerpt with deliberately changed spacing. Eight new published Commons clips add one natural control and one paired authored repeat each. Their download and qualification preceded scoring. The group comparison covers 104 deliveries, with one H.264 and one VP9 version of every case. [VCD2 data terms](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA)

Each local delivery contains 225 decoded pictures. Source excerpts were center-cropped to 16:9 where necessary and prepared on a 25 fps clock. The two lossy encodes use H.264 CRF 23 and VP9 CRF 31. Authored repeats vary from 23 to 34 native pictures. A useful result needs at least 0.50 intersection over union on each of the two ranges. The preregistered gate requires four useful camera copies and five useful Commons copies per codec, at most one additional control flag per codec, zero additional hold or spacing flags, and preservation of every useful exact result.

### Published activity and edit controls

The published-control rows use the selected nine seconds of each file with only format conversion. The Commons copyright terms permit commercial use under their respective conditions. The study publishes case outcomes and source credits. Media and altered copies remain local.

| Source | Author and terms | Start of excerpt |
| --- | --- | ---: |
| Interval Push-ups | Taco fleur, CC BY-SA 4.0 | 8 s |
| Golf swing practice, slow motion | Nesnad, CC BY 4.0 | 1 s |
| Rowing near Dingle Peninsula | Maoileann, CC BY-SA 4.0 | 2 s |
| Forward lunge training | Skhulile Mthiyane, CC BY-SA 4.0 | 8 s |
| Wobble board after being flicked | David Maisel, CC BY 3.0 | 2 s |
| Bench tennis | Evworo, CC BY-SA 3.0 | 1 s |
| Wind farm video | Wikideas1, CC0 | 2 s |
| Army fitness team readiness exercise | U.S. Army, video by Sgt. Brandon White, U.S. public domain | 3 s |

[Interval Push-ups](https://commons.wikimedia.org/wiki/File:Interval_Push-ups.webm) · [Golf swing practice](https://commons.wikimedia.org/wiki/File:Golf_swing_practice_-_Kanagawa_-_slow_motion_-_2023_June_13.webm) · [Rowing](https://commons.wikimedia.org/wiki/File:Rowing-1280187,_Dingle_Peninsula,_Co._Kerry,_Ireland.webm) · [Forward lunge](https://commons.wikimedia.org/wiki/File:Forward_lunge_training.webm)

[Wobble board](https://commons.wikimedia.org/wiki/File:Wobble_board_after_being_flicked.ogv) · [Bench tennis](https://commons.wikimedia.org/wiki/File:Bench_tennis.webm) · [Wind farm](https://commons.wikimedia.org/wiki/File:Wind_farm_video.webm) · [Army exercise](https://commons.wikimedia.org/wiki/File:Army_Fitness_Team_and_World%27s_Strongest_Man_Participate_in_Readiness_Exercise_o1HuhczOTL8.webm)

## Held results

| Outcome | H.264 | VP9 |
| --- | ---: | ---: |
| Useful camera copies, exact → combined | 0/6 → 4/6 | 0/6 → 4/6 |
| Useful published-source copies, exact → combined | 0/8 → 7/8 | 0/8 → 7/8 |
| Matched camera controls flagged | 0/30 | 0/30 |
| Published activity and edited controls flagged | 0/8 | 0/8 |
| Clock abstentions among the matched camera controls | 6/30 | 6/30 |

The comparison met every frozen requirement. In each codec one camera copy produced a range shorter than the required overlap, and one camera copy and the push-up copy had no qualifying range. Ordinary loops and repeated edits can produce a valid matching sequence. The real-action and editorial controls cover a small selection of activities. [Case-level aggregate](evaluation-panels/video-motion-sequence-360p-transfer-2026-11-12.json)

The independent replay checked every case using 23,400 native RGB hashes and timestamps, then independently recomputed exact and approximate pairs. The normal Full worker received the H.264 MP4 and VP9 WebM deliveries from held camera 9 through simulated checkout. Both reported **0.840–1.720 s** and **5.880–6.760 s**. All eight endpoint images matched the exact original pixels in the evidence and both signed packages replayed against their original uploads. The PDF and JSON used the same two time ranges for each codec. Chrome displayed all four 640-pixel source pictures in the plain evidence view and sought to **5.880 s** from its jump link. The static hold and published bench-tennis controls had no qualifying motion repeat. The changed-spacing control recorded an unavailable outcome.

## App coverage

The exact comparison still runs first. The similar-sequence fallback can now check the first 225 native pictures of unrotated 640 × 360 H.264 and VP9 in 8-bit 4:2:0. It needs verified 40 ms spacing across the complete prefix. The existing 320 × 180, 20 fps scope uses its original 180-frame bound and 50 ms spacing. Both signed packages from the earlier release replayed against their original files with the updated verifier. The packet shows two source time ranges, four source images and the supporting motion transitions. The observation carries zero AI-confidence weight.

## Reproduce

The public verifier checks the totals using the published case rows. Native replay uses the private study directory and its recorded delivery hashes. To regenerate the test, supply the qualified VCD2 v197 folder, the eight downloaded originals, and the retained private study folder from the earlier repeat evaluation for the digest-disjointness check.

```sh
cd backend
.venv/bin/python -B -m scripts.review_video_motion_sequence_360p_transfer ../docs/evaluation-panels/video-motion-sequence-360p-transfer-2026-11-12.json
```
