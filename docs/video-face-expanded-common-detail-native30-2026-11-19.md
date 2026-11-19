# Expanded native 30 fps short-swap comparison under blur

This follow-up uses the same fixed matched-detail comparison and the same
installed Full video stage as the preceding study. It provides twelve
staged opportunities to localize a staged three- or four-frame swap
with simultaneous blur across six retained source groups. The source files
have exact 30 fps native clocks. Source links, recorded donor keys, and
capture components separate the groups from each other and the earlier
brief-face, focus, blur, and common-detail cohorts. Other project exposure
and subject or pretraining overlap remain unknown. No source media leave
the local working archive.
[Preceding matched-detail study](video-face-common-detail-native30-2026-11-19.md)
· [Pinned dataset and terms](https://huggingface.co/datasets/godmodes/rtfs-10k/blob/bca865e675852666a426368b2f5614abf1c1926a/README.md)

Before any baseline result, the study salted the eligible source lineage
and proposed one edit in each of two native frame strata, starts 61–78
and 91–105. It balanced three- and four-frame widths, two retained swap
renderers, Gaussian and circular blur, and three blur strengths. Each
short placement created a clean swap, the matching ordinary blur, and that
swap altered with exactly the same blur. Each group also supplied one
untouched window and an ordinary hard cut to a different genuine recording.
The 48 deliveries use 120 exact native pictures, upright 720p AVC 4:2:0,
and a 30,000-tick time base. The archive authoring hashes and an independent
source reader verified 5,760 decoded delivery timestamps and all 42 native
picture timestamps in the twelve held edit windows.

We ran the enabled local Full stage once on every case. The optional
native three- and four-picture review was on. The three research arms
share the fixed blind detail-loss proposal schedule and use measured flank
landmarks to infer geometry through the short interval. The raw arm
compares the original crops; the old correction applies a fixed softening
explanation to its appearance score; the matched-detail arm softens the
sharper pictures enough to reduce quality differences before measuring
appearance. All arms use the same appearance point and preserve prior
qualifying ranges. Existing cuts and incompatible geometry remain excluded.
The reader counts both exact ranges and useful overlap at temporal IoU at
least 0.50. A miss is measured against all relevant incumbent Full ranges.
The gate requires two exact compound gains across at least two groups,
at most one added ordinary alert, at most one wrong added positive range,
and six ordinary files with usable views.

## Held outcome

| Comparison | Exact clean swaps / 12 | Exact blurred swaps / 12 | Ordinary alerts / 24 | Ordinary files with supplemental views | Wrong new positive locations | Frozen gate passed |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Enabled Full alone | 11 | 5 | 1 | — | — | — |
| Full plus raw recovered views | 11 | 12 | 8 | 14 | 0 | No |
| Full plus prior softening correction | 11 | 12 | 4 | 14 | 0 | No |
| Full plus matched spatial detail | 11 | 11 | 2 | 13 | 0 | Yes |

The enabled stage left 7 blurred swaps without a useful location across 4 capture groups. It left 7 without an exact location across 4 groups. The matched-detail arm added 6 exact compound locations across 4 groups, with one new ordinary alert. The supplemental readings preserved 17 detections already offered by the enabled stage.

An independent pass reconstructed 42 distinct native picture and aligned crop witnesses at the original source timestamps. Different interventions on one recording share that capture component. The added ordinary alert was the short circular-blur control at frames [105, 109), also the coordinate of a separate matched swap-with-blur. The component describes a physical appearance change and may flag ordinary editing. Exact timing describes the authored digital treatments. Performance on optical focus changes, unrepresented sources and general AI authorship remains open.

[Anonymized measured intervals and counts](evaluation-panels/video-face-expanded-common-detail-native30-2026-11-19.json)

```sh
python3 -I -S backend/scripts/review_video_expanded_common_detail.py \
  docs/evaluation-panels/video-face-expanded-common-detail-native30-2026-11-19.json
```
