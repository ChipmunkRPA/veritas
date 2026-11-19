# Matching crop detail for short face changes under blur

The locally enabled Full face review located all four clean swaps and three
of four swaps with simultaneous blur in this held trial. A comparison at
matched spatial detail recovered the remaining exact three-frame interval
without adding an ordinary alert. It and the previous softening correction
both recovered that interval. The frozen test required added locations in
two source groups, so this trial did not qualify either correction for
serving.

## Frozen comparison

The test used four exact-native 30 fps archive groups with known recording,
component and donor keys excluded from the preceding short-swap, focus and
blur studies. Each group supplied six newly authored 120-picture 720p AVC
deliveries on a 30 kHz clock. The roles were untouched, ordinary local
Gaussian blur, ordinary local circular blur, a clean publisher swap, the same
swap with the same Gaussian blur, and a hard cut between genuine captures.
The test contains two three-frame and two four-frame interventions for each
short-edit role. Source recordings remain private. Broader project exposure,
subject overlap and pretraining exposure are unknown. [Pinned publisher
source and terms](https://huggingface.co/datasets/godmodes/rtfs-10k/blob/bca865e675852666a426368b2f5614abf1c1926a/README.md)

The baseline was the existing local Full stage with the native three- and
four-frame option enabled. All supplements preserve its qualifying ranges.
They use the same bounded local-detail pair proposals and estimated middle
geometry from the preceding bridge work. The earlier correction applies the
frozen local-softening explanation after the raw appearance reading. The
new correction compares original aligned source crops at a common spatial
detail level before measuring appearance. It uses the weakest median detail
from the before, during and after groups, and only softens sharper crops
within a fixed Gaussian grid. Cases with insufficient detail or poor
matching support are unassessed by the supplement. All three arms use the
same fixed internal appearance point. The quality experiment draws on work
showing that blur and resolution affect face comparisons. [Blur and
illumination study](https://drum.lib.umd.edu/items/fbfbae5b-0b71-4239-88f7-083ae48f2b88)
· [WACV workshop study](https://openaccess.thecvf.com/content/WACV2024W/VAQ/papers/Bhatta_Impact_of_Blur_and_Resolution_on_Demographic_Disparities_in_1-to-Many_Facial_Identification_WACVW_2024_paper.pdf)

| Held comparison | Exact clean swaps / 4 | Exact swaps with blur / 4 | Ordinary alerts / 16 | Wrong added positive ranges |
| --- | ---: | ---: | ---: | ---: |
| Enabled Full stage | 4 | 3 | 0 | — |
| Full plus raw recovered views | 4 | 4 | 2 | 0 |
| Full plus the previously fixed softening correction | 4 | 4 | 0 | 0 |
| Full plus matched-detail comparison | 4 | 4 | 0 | 0 |

The supplements found usable comparison views in eight ordinary files. The
other eight do not establish the supplements' false-alert behavior. The
unfiltered arm flagged the matched ordinary Gaussian and circular edits
from the same source group as the recovered swap. Both corrections withheld
those two ordinary ranges. The recovered swap spans native frames
`[82, 85)` and retains source witnesses at 81, 83 and 85. An independent decoder
replayed nine unique original frame and aligned crop witnesses across the
comparisons. The seven baseline exact ranges leave only one opportunity for
a new location in this panel. A separate 27-file feasibility replay on earlier
29.97 fps development media retained all five previously found compound
gains and reduced four added ordinary alerts to one using matched detail.
That replay contributes no held outcomes.

## Next cohort

Freeze an expanded panel from additional groups selected by documented source
lineage and native clock, before any appearance readings. Use two distinct,
hash-selected edit placements per group, balance three- and four-picture
widths, and assign blur strengths across groups before encoding. Keep a
same-source clean swap, the same blur without the swap, and untreated and
hard-cut controls. A separate pre-score stratum can use source face size or
a fixed pixel-only measurement of local detail loss to include difficult
inputs. Record every admitted case and retain unassessed files in the
denominator. Run the enabled stage and both locked corrections once on the
whole panel. If the baseline misses too few compounds to assess the frozen
gain, report that ceiling with no replacement cases chosen from model output.

[Anonymized measured intervals and counts](evaluation-panels/video-face-common-detail-native30-2026-11-19.json)

```sh
python3 -I -S backend/scripts/review_video_common_detail.py \
  docs/evaluation-panels/video-face-common-detail-native30-2026-11-19.json
```
