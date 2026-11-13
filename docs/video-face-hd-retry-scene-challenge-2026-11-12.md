# Challenging brief face ranges with ordinary video cuts

## Fixed comparison

The app reviews 120 pictures in a supported 720p H.264 MP4 at exact 30 fps. Its brief face check first measures 8 Hz. When this pass finds a distinct brief range, the existing app records that the range falls outside the 720p publication profile. When it measures no distinct range, the app can make a guarded 15 Hz pass. This study tests trying the same guarded pass in the first situation as well, within the original deadline. The published range would still use the guarded pass, and any incumbent observations would be retained. The candidate has zero AI-confidence weight. [Current profile](video-face-two-returns-hd-2026-11-12.md) · [Earlier retry study](video-face-hd-sparse-retry-2026-11-12.md)

The earlier comparison delivered no clip to this retry branch. Here, eight disjoint pairs of genuine GRID recordings supply five controlled deliveries each. Two separate people occupy the same webcam tile for a brief conventional cutaway and return. The whole tile switches. Its fixed geometric surround supplies background texture outside the face and webcam picture. The controls include the unchanged layout, a same-person jump, a same-person digital punch-in, and a more compressed copy of the different-person cutaway. The three-second source footage is delivered in a four-second layout at 30 fps. The fixed recipe repeats existing 25 Hz pictures as needed and reflects the source at the ends. No intermediate pictures are generated. These authored controls test the behavior of the scene guard, and their counts describe this layout rather than the frequency of false alerts in ordinary uploads. [Cooke, Barker, Cunningham and Shao, GRID corpus, CC BY 4.0](https://zenodo.org/records/3625687) · [Source preparation and attribution](grid-transfer-acquisition-v130-method.md)

Seven separately matched RTFS groups supply original, publisher-swap and tandem-compressed copies. Original links and known donor keys exclude overlap with the recent short-range and retry panels. The project had earlier evaluation exposure to these archived groups. The original 30 Hz source is used at the matching indices, and independently qualified publisher swap pictures replace eleven pictures. There are two fixed half-open placements, [63, 74) and [85, 96). Each covers three sparse samples. The seven sources alternate the two retained swap families. The publisher declares CC BY-SA 4.0. Donor attribution and identity linkage are incomplete, so the public receipt contains aggregate counts and native timing examples only. [Publisher dataset and grant](https://huggingface.co/datasets/kontur-ai/rtfs-10k) · [Pinned media and permission review](video-common-source-preparation-v89.md)

The trial fixes the source roster, intervention timing, thresholds and candidate before either arm measures the clips. The isolated sparse screen checks branch plausibility without track inference. All 61 clips then run through the actual Full face path, including tracking and incumbent range deduplication. The retry runs only on the Full path's existing unpublished sparse-range disposition. The isolated screen and Full run are counted separately. A selected range needs at least half temporal intersection over union to count as a useful localization. The existing trial gate requires three additional useful ranges across two source groups, up to one original file with an added range, no ordinary cut files with an added range, and up to one misplaced retry range on edited footage. Added ranges count toward the gate even when the incumbent already reported a broader range on that file.

All 16 genuine cutaways and 13 of 14 known swap deliveries qualified in the isolated 8 Hz screen. Every isolated range overlapped the corresponding authored change by at least half. One tandem swap remained unassessed after a multiple-face detection. The remaining 31 deliveries yielded no distinct sparse range. The Full path makes a separate routing decision after checking overlap with its incumbent face range.

## Full results

| Delivery | Files | Full retry | Useful ranges or files flagged before | With retry |
| --- | ---: | ---: | ---: | ---: |
| Publisher swap | 7 | 1 | 6/7 useful | 6/7 useful |
| Tandem publisher swap | 7 | 2 | 5/7 useful | 5/7 useful |
| RTFS original | 7 | 0 | 0 flagged | 0 flagged |
| GRID unchanged layout | 8 | 0 | 0 flagged | 0 flagged |
| Genuine cutaway | 8 | 2 | 7 flagged | 7 flagged |
| Tandem genuine cutaway | 8 | 1 | 8 flagged | 8 flagged |
| Genuine same-person motion jump | 8 | 0 | 0 flagged | 0 flagged |
| Genuine same-person punch-in | 8 | 0 | 0 flagged | 0 flagged |

The Full path retried **three of sixteen** genuine cutaway deliveries across two speaker pairs. All three completed the 60-frame dense check and produced no additional range. The incumbent path had already highlighted an appearance change on fifteen of sixteen cutaways, including two of the three eligible for retry. That existing cue describes a visible change and carries zero authorship weight. No unchanged-layout, same-person jump or same-person reframe control received a range.

Three known swap deliveries from two source groups reached the retry. They were exactly the three missed localizations in this study. The incumbent chose a broader range over frames [55, 71), which overlapped each known [63, 74) edit by 0.421 at the native clock. All three completed the guarded measurement and produced no additional range. A first-encode genuine cutaway reached the same branch without an incumbent range and also produced no dense range. The retry stays experimental. [Held counts and native audit](evaluation-panels/video-face-hd-retry-scene-challenge-2026-11-12.json)

The Full comparison took 402.3 seconds, with 5.51 GiB peak process RSS and 12.84 MiB of encoded deliveries. The audit reopened 7,320 native pictures and rebuilt the six branch deliveries byte for byte. There were no newly selected crops or scene boundaries to verify.

## Local replay

Use the retained archives and a new private work directory. The freeze binds publisher and source receipts, detector assets and the serving comparator. The video comparator matches revision `d4937d8`. The audit reopens every delivered source picture and independently rebuilds every clip that reaches the Full retry. [Local face asset terms](video-appearance-plan-v107-method.md)

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze prepare screen score report; do
  backend/.venv/bin/python -B -m scripts.video_face_hd_retry_scene_challenge "$phase" "$run" || exit 1
done
```
