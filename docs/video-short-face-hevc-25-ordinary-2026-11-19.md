# Brief face evidence on ordinary 720p HEVC video

A prospectively sealed check reused the 25 fps HEVC carrier, matching
algorithm and internal point from the [held swap study](video-short-face-hevc-25-holdout-2026-11-19.md).
Nine additional native camera
groups came from an eligible pool of nineteen. The project excluded
known capture links, source components and donor lineages from the
first held test and the preceding related studies. The first nine salted
groups all qualified using only untouched original pictures. The
preflight required eight valid face samples and a stable single view
around one of four predeclared intervals. It checked original picture
geometry and texture before making any altered clip, without running
the face appearance comparison. The sampling procedure intentionally
enriches the cohort for ordinary material the check can assess.

Each camera supplied an unchanged excerpt and matched digital Gaussian
and circular blur over three or four frames. Five additional files splice
a new genuine camera at a known frame and exercise the cut guard
separately. A separate scene-transition intake on untouched pictures
identified zero natural cuts, so the cut evidence covers authored
straight edits between independent original recordings. Camera and
source group checks remained fixed before scoring.

| Ordinary condition | Files | Comparable views | Highlights with Full alone | With the supplemental pass |
| --- | ---: | ---: | ---: | ---: |
| Unchanged original excerpt | 9 | 0 | 0 | 0 |
| Local Gaussian blur | 9 | 9 | 0 | 0 |
| Local circular blur | 9 | 8 | 0 | 1 |
| Authored hard cut | 5 | 0 | 0 | 0 |

The comparison measured ordinary blur on all nine independently
selected groups and in 17 of 18 blur files. One circular blur produced
a highlight at original frames **[73, 76)**, **2.920–3.040 s**. Native
before, during and after views are frames **72, 74 and 76**. The ordinary
cut checker located the exact splice in all five cut files. The
supplement produced no cut highlight. The follow-up met its frozen
conditions of at least six ordinary groups with measurements, at most
one added ordinary highlight, and zero cut highlights. The unchanged
files supplied no candidate interval, which leaves them unassessed by
the optional short-blur comparison.

Independent decoding checked 3,840 original picture timestamps and
RGB frames. Original-file replay rechecked all three display anchors
for the ordinary blur highlight. The first held trial remains a
separate measurement of useful localization. This trial measures
ordinary outcomes within stable single-face source geometry and
digitally authored blur. Optical lens motion, other encoders and
cadences, and general detection accuracy remain unmeasured here.

The camera sources and replay implementations were separate from the held
swap trial. The project team ran both trials locally.

```sh
python3 -B -I -S backend/scripts/review_hevc_ordinary_confirmation.py \
  docs/evaluation-panels/video-short-face-hevc-25-ordinary-2026-11-19.json
```
