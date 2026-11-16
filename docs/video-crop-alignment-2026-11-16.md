# Align repeated footage to a more precise crop

The cropped-footage check keeps qualifying results from the fixed
search. After an empty fixed search it compares more precise centered
positions and crop geometry estimated from corresponding image details.
It confirms both candidates by checking the same sequence of picture
changes across two nonoverlapping ranges. The report shows both time
ranges, four original frames and the two raw source crops from the
wider range. Ordinary reframing can produce this observation. It
carries zero weight in the AI assessment.

## Alignment and temporal support

The centered comparison places five existing crop sizes symmetrically
around the center without rounding their edges to whole working pixels.
The second comparison matches local image details in every sixth
picture, estimates a common scale and translation, and refines it using
pixel intensity. Crop edge coordinates are recorded to
1/32 of a working pixel. Area averaging on the continuous crop
preserves the original decoded timeline. Both comparisons need
matching picture changes through at least half a second of moving
footage. Crop changes inside the candidate range, replay at another
speed, and other transformations can limit coverage. No new weights
are needed. [Regional video copy research](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01818.pdf)
· [Fine-grained video copy research](https://doi.org/10.1109/TPAMI.2025.3595625)
· [OpenCV alignment method](https://docs.opencv.org/doc/doxygen/html/dc/d6b/group__video__track.html)

This extends the existing `VERITAS_VIDEO_CROPPED_SEQUENCE_REVIEW` local
option. The source requirement stays a regular 25 fps H.264 MP4 with
8-bit 4:2:0 frames. The checked prefix is the first 225 frames at
640 × 360 or the first 200 at 1280 × 720. The signed record uses the
same source frame clock and the same crop packet ID. The original
video can reproduce the selected time ranges and pixels. Historical
fixed-crop signed packages still verify.
[Earlier fixed search](video-cropped-reuse-2026-11-16.md)

## Held clips and control edits

Three camera clips from the earlier study supplied development cases.
The held panel uses eight different VCD2 camera IDs in four scenarios,
plus two Commons recordings. The canyon clip contains a native camera
zoom and pan. The railway clip contains repeated visual structure and
moving subjects. Each group supplies unchanged and recompressed
controls, in-place and moving crops, held cropped frames, and repeats
without a crop. The staged positives cover centered and off-center
crops with both earlier and later reuse and separate second encodes.
Source hashes, transformations, exact intervals and the two comparison
stages are retained in the receipt.
[VCD2 data terms](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA)
· [Bright Angel Flash Flood by Grand Canyon NPS / K. Hensel](https://commons.wikimedia.org/wiki/File:Bright_Angel_Flash_Flood_(55504573056).webm)
· [Railway Crossing by WeenyBeany](https://commons.wikimedia.org/wiki/File:Railway_Crossing_In_Action_with_Train_Passing_Through.webm)

At each resolution, 70 staged deliveries contain a cropped repeat
and 60 are controls. A useful location requires at least 0.50
intersection over union for each range. The comparisons use exactly
the same decoded frames. The two subpixel methods run separately
before combining them with the fixed search.
[Numeric panel](evaluation-panels/video-crop-alignment-2026-11-16.json.gz)

| Resolution | Fixed search | Centered alignment | Landmark alignment | Combined | Combined alerts on controls |
| --- | ---: | ---: | ---: | ---: | ---: |
| 360p | 15/70 | 6/70 | 25/70 | 33/70 | 0/60 |
| 720p | 17/70 | 9/70 | 38/70 | 43/70 | 0/60 |
| Total | 32/140 | 15/140 | 63/140 | 76/140 | 0/120 |

The combined rule adds 44 useful deliveries. At the standard
centered size, it finds 15/20, up from 6/20. At fractional centered
sizes it finds 17/40. For unsampled off-center crops it finds 18/40.
The 40 successes with exact endpoints include both delivery profiles.
The 76 combined matches identify the wider occurrence correctly.
The largest measured crop-edge error is 1.3 pixels on the 192 × 108
working grid. These counts measure repeated footage and these encoding
profiles. The combined check still misses 64 of the 140 staged repeats.

The independent reader checked all 76 matched deliveries against
their native timestamps. It verified 744 endpoint appearances across
the three comparisons and the combined result. The median added
comparison time on this Mac was 0.44 s per 360p clip and 0.34 s per
720p clip. Those times cover the candidate calculation after decode.

## Local app verification

Six simulated USD 9.99 checkouts isolated the repeat checks in the
ordinary worker and report pipeline. No jobs queued before payment.
The 360p centered case displayed source ranges at 1.000–1.880 s and
5.960–6.840 s. The 720p landmark case, which was encoded twice,
displayed 1.080–1.880 s and 5.640–6.440 s. Both in-place crop
controls stayed clear. The browser, PDF, and JSON showed the same
original frames and crops. Signed verification matched the packages
to each original video and rejected a changed crop coordinate.
Both paired runs kept the same overall AI assessment.
[Local replay receipt](evaluation-panels/video-crop-alignment-app-2026-11-16.json)

## Replay

Use Python's standard library to count the fixed, centered, estimated
and combined results. With the delivered clips available, this also
verifies each reported timestamp using FFprobe. To regenerate those
deliveries, restore each recording from the ledger and add its local
`path` to a source array. The alignment freeze pins the renderer's
input plan, the candidate implementation, and the OpenCV version.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
for profile in 360 720; do
    "$runner" -B -m scripts.video_cropped_reuse_panel build "$run/held-$profile" "$sources" "avc-$profile" --alignment-plan
    "$runner" -B -m scripts.video_crop_alignment_panel lock "$run/held-$profile" "$run/held-$profile-freeze.json"
    "$runner" -B -m scripts.video_crop_alignment_panel score "$run/held-$profile" "$run/held-$profile-freeze.json" "$run/held-$profile-result.json"
done
receipt=docs/evaluation-panels/video-crop-alignment-2026-11-16.json.gz
python3 -B -S -m scripts.review_video_crop_alignment "$receipt" --media "$run"
npm --prefix frontend run build
"$runner" -B -m scripts.video_cropped_reuse_app "$run" "$local_app" "$run/held-360-result.json" "$run/held-720-result.json" --browser
```

Use a filesystem with hard-link support for `$local_app`.
