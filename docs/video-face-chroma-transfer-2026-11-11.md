# Local face-swap color-step comparison

A face swapped throughout a clip can leave no original segment for comparison. This test looked for a persistent color change between the inner and outer face on both sides, using exact native frame samples. The October 2026 illumination study found no consistent benefit specific to the illumination effect it tested. That motivated testing this separate color measurement at a fixed point on held-out clips. [Illumination study](https://arxiv.org/html/2610.11706v1)

## Method

The measurement sampled 11 horizontal rows from 43% to 72% of the detected face-box height. It compared four bands at each of four ellipse-shaped offsets from the center of the box. Each band used the median of three adjacent native RGB pixels after conversion to luminance and two chroma channels. The calculation subtracts the local flank slopes from the central change, requires the two cheek sides to change in a similar chroma direction, and downweights large accompanying luminance changes. A frame keeps the strongest of the four measurements. A clip uses the median over six fixed sample positions, with at least four measured faces and a minimum native face-box side of 64 pixels.

The threshold was fixed at 0.8329082034198929, the 90th percentile of the nine assessed development originals. Equality did not flag. An authored bilateral color step with constant luminance scored 17.0. The flat, smoothly varying, luminance-only and opposing-color controls each scored zero.

The local runner verified the native source hashes, timestamps, RGB and face boxes against the previous classifier receipts. The prospective run kept the method and point fixed. It included eight native HD and 12 matched-delivery source groups with no source IDs shared with the earlier trial. Both cohorts used the same two generator families as the earlier comparison. The fresh run checked 360 exact native frames across 60 clips and made zero additional neural calls. [Prior paired trial](video-full-clip-face-swap-transfer.md) · [Aggregate and configuration](evaluation-panels/video-face-chroma-transfer-2026-11-11.json)

## Results

| Cohort | Swap flags | Original false flags | Unassessed clips |
| --- | ---: | ---: | ---: |
| Development | 1/20 | 0/10 | 3/30 |
| Historical confirmation | 0/20 | 0/10 | 3/30 |
| Prospective native HD | 2/16 | 1/8 | 0/24 |
| Prospective matched delivery | 4/24 | 2/12 | 0/36 |

The historical transfer also produced zero flags on four genuine camera clips and the two clips in the publisher pair, which uses a synthetic base. The first camera probe reached the 25-second bound on a 491 MB file. A repeat extended the camera probe allowance to 70 seconds and completed every case at the same inspection point.

On the prospective clips, two swap flags and three original false flags were additional to the flags from the two crop classifiers at their frozen individual points. Both additional swap flags came from matched delivery. The color measurement remains experimental. The public receipt supports aggregate review. Private source files, source IDs, per-case responses, frame timestamps and face boxes stay local.
