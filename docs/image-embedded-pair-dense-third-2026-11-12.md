# Dense native-pixel evidence for a resized third view

The dense verification trial withheld seven of 16 remaining ambiguous comparisons. The current 768 × 768 size gained one withheld comparison out of five. The other two sizes gained three each. The trial preserved the 70 exact two-area masks on eligible controls. All sizes fell short of the frozen criterion, so the dense guard stays in research. [Frozen scoring and supplemental readback](evaluation-panels/image-embedded-pair-dense-third-2026-11-12.json)

## Why test dense evidence

A diagnostic replay of all 37 resized-third misses from the earlier canvas study found usable transforms in every case. The estimated scale differed from the authored scale by at most 0.003 and estimated rotation stayed within 0.31 degrees. Feature positions fell short of the existing target-area spread requirement in 36 cases, the source-area spread in 26, and the local correlation support in 11. The conditions can overlap. Some small or smooth photos yielded accurate alignments from features concentrated in one part of the image. [Canvas study](image-embedded-pair-canvas-transfer-2026-11-12.md)

The prospective method keeps the current geometric proposal. It adds a guard only when the current check reports a difference. The guard projects the first comparison area onto a separate possible third view. It scores 16 × 16 grayscale patches on a fixed 19-pixel grid, skips low-texture positions and pixels covered by the actual difference, and requires correlation of at least 0.97 on at least 16 positions in six spatial cells and 55% of the usable source patches. The candidate region must stay inside the canvas and clear both comparison areas. The verified difference masks remain exact RGB measurements.

## Frozen held comparison

The study used 12 distinct photo groups separate from the earlier canvas panel. The source cache contains attributed CC BY 2.0 images with prior project exposure. The controlled native changes are ordinary copy-and-paste edits. Each size has 60 third-view challenges and 36 controls. The third view uses an independently selected placement and is resized to 0.62×, 0.71×, 0.87×, 1.13×, or 1.21×. Different groups use Lanczos or bicubic resampling, and four groups per size also apply a JPEG round trip to the third view. Control pairs include a real local change without a third, the same change beside an unrelated resized photo, and identical duplicate panes. [Source and attribution checks](image-fresh-source-cache-v39-results.md)

The preregistered requirements included at least three remaining ambiguities per size, an 80% reduction in those reports, at least ten exact masks in each valid control category, no additional withholds on controls, and no candidate step longer than 1.5 seconds. The seal is `ff0818f6cc30d5af7cc2a64567c469e4b02ecec7896955b287500d6f0d4f9655`.

| Canvas | Third-view reports before guard | Reports after guard | Exact valid control masks preserved | New control withholds |
| --- | ---: | ---: | ---: | ---: |
| 768 × 768 | 5/60 | 4/60 | 24 | 0 |
| 1024 × 768 | 5/60 | 2/60 | 22 | 0 |
| 1280 × 720 | 6/60 | 3/60 | 24 | 0 |

The first native audit stopped at its whole-rectangle requirement. Six of seven estimated comparison rectangles overlapped the authored third view by 85.7–89.5%, below the 90% requirement. The supplemental audit used unchanged scored inputs and verified all 497 recorded patch correlations. It traced 482 patch centers to the authored third view and preserved the six whole-rectangle failures in the receipt. Six of the seven added withholds came from one of the 12 photo groups. That concentration and the failed frozen gate limit the evidence for a broader change.

## Reproduce

From the repository root, use the retained source archives and a new local directory. The scorer freezes its code and exact pixel inputs. The supplemental audit independently remaps the stored source coordinates and reproduces the arithmetic of the results. The published receipt excludes source images, full geometry, and individual patch positions.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.image_embedded_pair_dense_third freeze "$run"
backend/.venv/bin/python -B -m scripts.image_embedded_pair_dense_third score "$run"
backend/.venv/bin/python -B -m scripts.audit_image_embedded_pair_dense_third "$run"
python3 -I -S backend/scripts/review_image_embedded_pair_dense_third.py "$run/supplemental-public.json"
```
