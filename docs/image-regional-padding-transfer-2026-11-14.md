# Padding outside the source image and small JPEG edit localization

This comparison used the same 186 JPEG75 deliveries from 30 photo groups
as the [small-region repeat study](image-small-region-repeat-2026-11-14.md).
The 60 edited deliveries cover Firefly, FLUX Fill, Stable Diffusion 2 and
LaMa. The 126 ordinary controls include unchanged sources and local
processing. This was a follow-up on a previously measured panel.

The existing checkpoint and input geometry were frozen. The two variants
replaced black outside the submitted photo with either extended edge values
or reflections. A binding at the model's patch input checked the exact
expected patch tensor. For each variant, all source pixels and their
placement were identical to the current path. The combined arm preserved
all existing regions and added a candidate only when at least two thirds
of both variant components overlapped. The model response cutoff, image
area gate and reliable component rules were held fixed.

## Observed results

| Input outside the source image | Useful edit location | Misplaced location on an edited file | Highlight on an ordinary control |
| --- | ---: | ---: | ---: |
| Black, current path | 0/60 | 9/60 | 20/126 |
| Extended edge | 0/60 | 7/60 | 13/126 |
| Reflection | 0/60 | 18/60 | 44/126 |
| Current path plus agreement | 0/60 | 9/60 | 20/126 |

A useful location required intersection over union of at least 0.30 and
changed-pixel recall of at least 0.50. None of the variants met the study
criterion of three additional useful groups across two editor families.
The extended-edge arm removed seven LaMa control highlights and two
misplaced LaMa edit highlights. The corresponding change may merit a
separate specificity test on fresh groups containing already detectable
edits. Reflection added ordinary highlights in ten groups relative to
the current path.

The median absolute response change averaged over each image was 0.00556
for extended edges and 0.115 for reflection. For reflection it was 0.246
in the bottom and right 64-cell strips and 0.0654 in the center after a
128-cell inset. Removing the area gate from either variant revealed the
same feathered FLUX Fill edit already found when bypassing the current
gate. These results apply to localization under this fixed JPEG profile.
The alternate padding methods remain experimental.

## Replay

The [numeric ledger](evaluation-panels/image-regional-padding-transfer-2026-11-14.json)
has the exact input hashes, component counts and source-group results.
The recount needs standard-library Python and no source images or weights.

```sh
PYTHONPATH=backend python3.12 -B -S -m scripts.image_regional_padding_transfer review \
  docs/evaluation-panels/image-regional-padding-transfer-2026-11-14.json
```

Native replay needs the TGIF/TGIF2 source archive, both BG-REAL rosters
from the parent study and the pinned regional checkpoint. Run `freeze`
and then `score` with the same new directory and both BG-REAL roster
directories in order after `--bg`. The runner checks that the parent
source roles and the reproduced existing regions match the earlier ledger.
