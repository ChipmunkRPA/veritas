# Footage reused with a tighter crop

This check compares two time ranges in one video after accounting for
fixed crop and scale changes. The evidence identifies both ranges and
shows their original frames, plus exact source crops from the wider
view. Ordinary editing can produce the same observation. It carries
zero weight in the overall AI assessment.

## Method and scope

The existing comparisons look for repeats at the same framing, reversed
frames, a two-to-one speed change, or a left-to-right flip. This check
samples 45 crop placements at five retained frame sizes between 70%
and 90% of the original width and height. It compares each crop and
full frame using small luminance grids, requires a substantially
different direct view, and checks that motion changes match through a
continuous pair of nonoverlapping ranges. Both ranges must use the same
playback speed, with at least half a second of suitable footage.
It uses no model weights. See [spatiotemporal copy matching](https://www2.cs.sfu.ca/~mhefeeda/Papers/mmsys12-Cameron.pdf)
and [fine-grained copy localization](https://doi.org/10.1109/TPAMI.2025.3595625)
for related approaches.

The development setting is `VERITAS_VIDEO_CROPPED_SEQUENCE_REVIEW=true`.
The check inspects the first 225 frames at 640 × 360 or the first 200
frames at 1280 × 720. It requires a verified regular 25 fps clock and
upright 8-bit H.264 MP4 using 4:2:0 chroma. The exact original frame
timestamps label the evidence. Existing repeat checks run first.

## Matched source results

Three Commons recordings supplied development examples. The held
comparison uses eight other camera recordings from four VCD2 scenarios
and eight distinct camera IDs. Each recording supplies matched native
360p and 720p H.264 deliveries. The source archive, recipe, clip digest
and labeled ranges appear in the numeric ledger. A useful location must
overlap each corresponding label by at least half of the union.
[Camera source terms](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA)
· [Delivery ledger](evaluation-panels/video-cropped-reuse-2026-11-16.json.gz)

| Compared delivery | Existing locations | Crop locations, 360p | Crop locations, 720p |
| --- | ---: | ---: | ---: |
| Offset crop at a sampled position | 0/16 | 7/8 | 7/8 |
| Offset crop after a second encode | 0/16 | 6/8 | 7/8 |
| Centered crop between tested pixel boundaries | 0/16 | 0/8 | 0/8 |
| Crop between tested positions and scales | 0/16 | 0/8 | 0/8 |
| Added alerts on controls | — | 0/48 | 0/48 |

The 27 located deliveries come from seven camera recordings. Both
ranges exactly match the known endpoints in 15 deliveries. A separate
FFprobe readback verified each of the 108 reported timestamps on the
original clip. Controls include unchanged clips, ordinary in-place
crops, moving crops, held cropped frames, straight repeats, and
second encodes of the unchanged clips. The existing check retained
its seven correct straight-repeat observations. The centered and
between-grid misses limit coverage of small changes in placement,
even within the tested zoom amounts. The sample tests staged copy
reuse on camera videos. Broader delivery and content coverage remain
unmeasured.

## Local app verification

Six simulated USD 9.99 checkouts exercised the ordinary analysis
pipeline. No job queued before payment. The 360p direct encode and
720p second encode produced their expected pair with the check
enabled. Both in-place crop controls stayed unflagged. The browser
displayed both ranges, four original frames and two source crops
for each match, and playback sought to the same source time. The
evidence API, JSON and PDF agreed. The signed packages verified
against the original files, and altered crop coordinates failed
validation. The overall AI assessment agreed with and without the
new check in the two paired runs. [Local app receipt](evaluation-panels/video-cropped-reuse-app-2026-11-16.json)

## Replay

Use the published receipt to recount locations without any model
packages. To rebuild the clips, restore the licensed sources at the
verified hashes and add a local `path` to each source row. The
detector source is pinned in each delivery plan.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1
receipt=docs/evaluation-panels/video-cropped-reuse-2026-11-16.json.gz
python3 -B -S -m scripts.review_video_cropped_reuse "$receipt"
runner=backend/.venv/bin/python
"$runner" -B -m scripts.video_cropped_reuse_panel build "$run/held-360" "$sources" avc-360
"$runner" -B -m scripts.video_cropped_reuse_panel build "$run/held-720" "$sources" avc-720
"$runner" -B -m scripts.video_cropped_reuse_panel score "$run/held-360" "$run/held-360.json"
"$runner" -B -m scripts.video_cropped_reuse_panel score "$run/held-720" "$run/held-720.json"
python3 -B -S -m scripts.review_video_cropped_reuse "$receipt" --media "$run"
npm --prefix frontend run build
"$runner" -B -m scripts.video_cropped_reuse_app "$run" "$local_app" "$run/held-360.json" "$run/held-720.json" --browser
```

Use a filesystem with hard-link support for `$local_app`.
