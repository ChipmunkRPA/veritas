# Testing a learned wavelet map on small JPEG edits

SAFE is a lightweight image classifier trained to distinguish generated
images using diagonal wavelet features. This test checks whether the same
learned features can locate a small change when the surrounding pixels are
preserved. It uses the publisher's 5.8 MB checkpoint from the Apache-2.0
release, verified by SHA-256 and read with a weights-only loader. The
evaluation implements the active inference path using existing PyTorch,
Torchvision, and PyWavelets operations. [Primary study](https://arxiv.org/abs/2408.06741)
· [Pinned release and checkpoint](https://github.com/Ouxiang-Li/SAFE/tree/4e998724651b227def64f5be0cd60c0aa1552c35)
· [Pinned license](https://github.com/Ouxiang-Li/SAFE/blob/4e998724651b227def64f5be0cd60c0aa1552c35/LICENSE)

Apply the model at the uploaded resolution. Read the model's trained
linear head at each spatial cell before image-wide averaging. The mean
of those local logits reproduces the publisher's classifier output.
A separate parity check compared the wavelet coefficients and final
class margin with the publisher's implementation at 224 × 224,
405 × 321, and 768 × 512. Maximum observed differences were
1.20 × 10⁻⁷ for the wavelet coefficients and 4.77 × 10⁻⁷ for the class
margin.

Each proposed square compares the mean trained class margin inside
against its surrounding ring. The score is the smaller of the internal
margin and its increase over the ring. Four square sizes span 40 to
136 submitted pixels. The app's separate existing localization checks
run on the same exact delivered JPEGs, with the repeated regional check
at quality 90 where it applies. A useful candidate covers at least half
the exact pre-encoding changed pixels and has at least 0.30 intersection
over union.

Fourteen retained scene-text source groups set the cutoffs. The cutoffs
take the second-largest per-source ordinary maximum among files the
existing checks left clear. The two points were locked independently
for JPEG quality 75 and quality 90 before held scoring. Each permits
one additional ordinary source group in calibration. The best rectangle
missed the required overlap on all 56 calibration edits before the
cutoff was applied.

The new scene-text holdout reserves the next 24 source sets from the
pinned TextEraseBench release and retains 15 after the same source and
mask checks used in the earlier comparisons. Those source images are
separate from the 30 previously prepared sources by file and decoded
image SHA-256. The published Nano Banana 2 backgrounds are applied only
inside the annotated area, with sharp and feathered variants. The
underlying source images may have unknown origin. The separate retained
holdout uses 12 BG-REAL camera sources with LaMa and MI-GAN removals.
Controls include unchanged images and matched local blur, tone, copying,
and conventional fills. [Scene-text and attribution terms](https://huggingface.co/datasets/QinmingZhou/TextEraseBench/blob/cf94f7eaffc654146cba02fe2f704cc1a6e4998d/README.md)
· [BG-REAL photo and editor terms](image-jpeg-frequency-bands-2026-11-17.md#held-photographs-and-test-controls)

## Local results

The frozen quality-75 and quality-90 cutoffs were 11.487542 and
15.783585 in class-margin units.

| Held panel | JPEG quality | Useful locations, existing → combined | Ordinary flags, existing → combined | Misplaced candidate regions |
| --- | ---: | ---: | ---: | ---: |
| Fresh scene-text removals | 75 | 0 → 0 of 30 | 9 → 31 of 75 | 13 |
| Fresh scene-text removals | 90 | 0 → 0 of 30 | 10 → 10 of 75 | 0 |
| Retained BG-REAL removals | 75 | 0 → 0 of 48 | 16 → 27 of 120 | 4 |
| Retained BG-REAL removals | 90 | 4 → 4 of 48 | 10 → 10 of 120 | 0 |

The top-ranked rectangle failed to locate any of the 156 held edits
even before either cutoff. An exhaustive search using the exact edit
masks found that the allowed geometry could locate 40 of 60 scene-text
edits and all 96 BG-REAL edits. The remaining scene-text edits exceed
what this geometry can recover. The 33 additional ordinary alerts include six
unchanged deliveries and come from seven source groups. These outcomes
leave the candidate offline.

The application's prior-grid check was separately replayed on all 105
fresh quality-90 files and produced no additional location. This matches
the earlier measurement on the retained BG-REAL bank. [Prior-grid comparison](image-local-spectral-peaks-2026-11-18.md#local-results)

An independent recount verified the source-file and truth-mask digests,
rectangle limits, and integer overlap for all 742 calibration and held
deliveries. It also verified the new source roster against both previous
scene-text intakes.

[Calibration ledger](evaluation-panels/image-safe-dense-freeze-2026-11-18.json.gz)
· [Held scene-text ledger](evaluation-panels/image-safe-dense-scene-2026-11-18.json.gz)
· [New source acquisition record](evaluation-panels/image-safe-dense-scene-intake-2026-11-18.json.gz)
· [Held BG-REAL ledger](evaluation-panels/image-safe-dense-bg-2026-11-18.json.gz)

## Reproduce

Retrieve the pinned publisher checkpoint. Restore the original scene-text
bank and the 12-photo BG-REAL transfer with the existing delivery recipes.
For the fresh scene-text input, combine the `cases` and `dispositions`
from the two earlier scene-text intake files, preserving their common
`revision`. Use that file as `--previous-intake` and start at 48 in the
same pinned publisher inventory. Keep the prepared manifests unchanged
when replaying the existing JPEG checks.
[Earlier scene text and pinned inventory](image-jpeg-pscc-localization-2026-11-17.md#reproduce)
· [Second scene-text intake and BG-REAL preparation](image-masked-reconstruction-2026-11-17.md#reproduce)

From `backend`:

```sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=.venv/bin/python
module=scripts.image_safe_dense_transfer
python3 - "$first_intake" "$second_intake" "$previous_scene_intakes" <<'PY'
import json
from pathlib import Path
import sys
first, second, output = map(Path, sys.argv[1:])
a, b = (json.loads(path.read_bytes()) for path in (first, second))
assert a['revision'] == b['revision']
combined = {'revision': a['revision'], **{key: a[key] + b[key] for key in ('cases', 'dispositions')}}
output.write_text(json.dumps(combined, sort_keys=True))
PY
"$runner" -B -m scripts.image_text_erase_localization_inputs prepare "$index" "$new_scene" \
  --start 48 --previous-intake "$previous_scene_intakes"
"$runner" -B -m scripts.image_region_rank_reference "$new_scene/deliveries" "$scene_baseline" --repeat-jpeg90
"$runner" -B -m "$module" freeze "$checkpoint" "$development" "$development_baseline" "$freeze" \
  --held "$new_scene/deliveries" --held "$bg" --held "$bg_fills"
"$runner" -B -m "$module" validate "$checkpoint" "$freeze" "$scene_result" \
  --banks "$new_scene/deliveries" "$scene_baseline"
"$runner" -B -m "$module" validate "$checkpoint" "$freeze" "$bg_result" \
  --banks "$bg" "$bg_baseline" --banks "$bg_fills" "$bg_fills_baseline"
python3 -B -S -m "$module" review "$freeze" "$scene_result"
python3 -B -S -m "$module" review "$freeze" "$bg_result"
```
