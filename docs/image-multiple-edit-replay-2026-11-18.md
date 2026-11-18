# Two separate pixel edits in one JPEG

Forgeries can contain several independent edits. One large highlight can
overlap all of them and still give a reviewer little guidance. This test
measures each edit separately. It borrows the separate-target evaluation
idea from process-aware localization research, without using its code,
datasets or weights. [CVPR Findings paper](https://openaccess.thecvf.com/content/CVPR2026F/html/Zhu_Revisiting_Image_Manipulation_Localization_under_Realistic_Manipulation_Scenarios_CVPRF_2026_paper.html)

The next 24 entries from a pinned publisher reserve yielded 14 usable
source sets. The files passed the prior source-identity and framing checks.
Each staged JPEG contains two separated changes on the same framing, a
published AI text removal placed inside its local region and a conventional
elliptical pixel copy. The controls are an unchanged JPEG export and a
second JPEG encoding of that export. Those three deliveries belong to one
source group. The publisher's generated background is used only inside
its marked area. The original source picture's provenance can be unknown.
[Pinned source and terms](https://huggingface.co/datasets/QinmingZhou/TextEraseBench/blob/cf94f7eaffc654146cba02fe2f704cc1a6e4998d/README.md)

Both methods receive an earlier lossless PNG of the same image. The
existing method establishes alignment and reports localized differences.
The experimental alternative re-encodes the earlier PNG using the supplied
JPEG tables, compares decoded pixels, and requires exact agreement on at
least 85% of aligned blocks. The qualification also requires detailed,
unique exact anchors across the frame. It first checks whether the entire
file matches one or two ordinary encodings. The two proposal sets were
also scored together. Each reported region could match at most one
separately measured edit, with at least 0.30 intersection over union and
0.50 edit coverage. Only the exact pre-export changed pixels count as
target area. The scene-text and copy masks were kept separate.

| Measure | Existing earlier-copy check | Exact delivery candidate | Combined |
| --- | ---: | ---: | ---: |
| Edit locations found | 27 / 28 | 21 / 28 | 27 / 28 |
| Files with both changes located | 13 / 14 | 10 / 14 | 13 / 14 |
| Unassessed edited files | 0 / 14 | 3 / 14 | 0 / 14 |
| Extra or misplaced proposals on edited files | 2 | 9 | 32 |
| Ordinary exports flagged | 0 / 28 | 0 / 28 | 0 / 28 |

The alternative located no additional edit. It stays outside the app.
The counts apply to same-framing JPEG quality-75 files with a matching
earlier PNG and two staged edits. The ordinary controls cover repeated
encoding under the tested settings. A measured pixel difference can come
from conventional editing and carries zero weight in AI authorship.

[Frozen inputs and matching rule](evaluation-panels/image-multi-region-freeze-2026-11-18.json) ·
[Aggregate replay results](evaluation-panels/image-multi-region-replay-2026-11-18.json)

## Replay locally

Rebuild from the [pinned scene-text inventory and prior intakes](image-pscc-native-tiles-2026-11-18.md#reproduce).
Append the fourth intake, the one starting at 72, to the combined
prior intake file with the same `revision`, `cases` and `dispositions`
structure. Keep the image files and exact masks local. The published
result contains counts and hashes. The independent comparison can read
the native receipts with the Python standard library alone.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
runner=backend/.venv/bin/python
"$runner" -B -m scripts.image_text_erase_localization_inputs prepare "$index" "$source" \
  --start 96 --previous-intake "$prior_intakes"
"$runner" -B -m scripts.image_multi_region_replay prepare "$source" "$run"
"$runner" -B -m scripts.image_multi_region_replay measure "$run" "$result"
python3 -B -I -S backend/scripts/independent_review.py compare-events \
  "$run/manifest.json" "$run/truth.json" "$run/current.json" "$run/exact_delivery.json" \
  "$aggregate" --match .30 .50 --image-mask-dir "$run/masks"
```
