# Transfer of brief face evidence to 720p HEVC at 25 fps

The local brief-face comparison previously measured 120 native 720p H.264
pictures at 25 fps. This prospective trial kept the existing internal
appearance point and fixed image, face and shot checks. It changed only
the admission to the supplemental pass. The tested HEVC profile has one
square-pixel, upright, 8-bit 4:2:0 hvc1 video track with exactly 120 native
pictures, zero timeline origin, a 90 kHz source clock and 3,600 ticks
between pictures. The independent MP4 atom check reproduces the complete
presentation from the local source before face scoring.
[Apple's HEVC capture documentation](https://support.apple.com/en-us/116944)

Five camera and swap groups were chosen before any scoring, independently
of the listed earlier matched-detail, brief-face and compound studies by
known capture links, source components and donor records. The source archive
still had prior general project exposure, and identity overlap beyond those
records is unknown. Two source groups were already 720p and three used an
area downsample from 1080p. All five retained the exact native 25 fps picture
ordinals. The archived swap pictures use two publisher renderers. The local
test changed no weights and keeps all source recordings private.
[RTFS-10K publisher and terms](https://huggingface.co/datasets/godmodes/rtfs-10k)

Each group supplied an unchanged camera excerpt, an ordinary hard cut, a
three- or four-picture light change, one staged local blur, a brief swap,
and the same swap with precisely the same blur. Paired ordinary blur and
swap-plus-blur treatments also went through the shipped H.264 profile with
identical pre-encode pictures. The tested HEVC encodes used one fixed
encoder configuration and the app's exact Full face analysis stages.
The new pass would add at most one location, preserving earlier findings
and excluding the same shot boundaries.

## Held result

| Condition | Eligible HEVC files | Exact locations or alerts with Full alone | With supplemental pass | Comparable face views |
| --- | ---: | ---: | ---: | ---: |
| Brief swap with the matched blur | 5 | 0 | 2 | 2 |
| Brief swap without added blur | 5 | 0 | 0 | 0 |
| Matched ordinary local blur | 5 | 0 | 0 | 2 |
| Ordinary local light change | 5 | 0 | 0 | 0 |
| Unchanged camera excerpt | 5 | 0 | 0 | 0 |
| Ordinary hard cut | 5 | 0 | 0 | 0 |

The supplemental pass added two exact compound ranges in two groups with
zero wrong locations. Both ends preserved the planted source-frame clock.
The first held example spans frames **[77, 81)**, **3.080–3.240 s**, with
original frames **76, 79 and 81**. The second spans frames **[97, 100)**,
**3.880–4.000 s**, with original frames **96, 98 and 100**. The H.264
comparison also localized two of five paired compounds. One exact range
overlapped across carriers and each carrier found one other group. One of
five paired ordinary H.264 blurs had comparable views without a finding.

None of the 20 ordinary HEVC clips added a highlight, and the shot review
observed all five true cuts. The detail comparison had usable views in only
two of 20 ordinary files. Sixteen ordinary files offered no shortlist
proposal and two had proposals without measurable views. The frozen gate
needed four measured ordinary files, so this trial did not qualify serving
expansion. The two exact gains motivate a separate specificity comparison
with strong evidence of assessable ordinary views.

Independent FFprobe and RGB readback verified 4,800 exact delivered frame
timestamps and decoded pixels and the ten matched pre-encode sequences.
Source replay verified 37 exact native measurement inputs and all 12
display anchors across the four marked deliveries. The anonymous numeric
receipt includes exact original-frame ordinals, no source/subject
identifiers, recordings, image crops or face geometry.

```sh
python3 -B -I -S backend/scripts/review_short_hevc_matched_detail.py \
  docs/evaluation-panels/video-short-face-hevc-25-holdout-2026-11-19.json
```
