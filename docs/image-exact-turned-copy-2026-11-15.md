# Locating exact rotated copies inside a PNG

Chipmunk Verify can locate small areas in one PNG that have identical RGB
pixels after a quarter turn or half turn. It shows both crops as they
appear in the submitted file, their positions on the original, and the
rotation that aligns Area B with Area A. The Standard PDF shows the two
native crops and the measured differences in their surroundings. The
JSON and signed evidence package use the same positions and comparison.

Repeated graphics, conventional edits and compositing can produce this
pattern. The comparison carries zero weight in the overall AI assessment.
The two areas' roles and the direction of any edit remain undetermined.

## Tested scope

The local trial covers upright static 8-bit RGB PNG files with sides
from 384 to 768 pixels. Each matched area has sides from 16 to 56 pixels.
It runs when the same-orientation small-copy review found no qualifying
pair. Native RGB signatures propose offsets, then direct comparison of
each pixel confirms the core. The core needs texture and at least 64
different RGB values. The three-pixel surroundings must differ on at
least three sides after alignment. If two distinct pairs qualify, the
check withholds the result. It can miss copies that were resampled, color
adjusted, recompressed or blended throughout. It excludes lossy JPEG,
other rotations and unsupported PNG profiles.

The local combination is
`VERITAS_IMAGE_SMALL_PNG_COPY_REVIEW=true` and
`VERITAS_IMAGE_TURNED_PNG_COPY_REVIEW=true`. Turning off the second
setting restores the prior comparison for new analyses. Stored evidence
from either version keeps its original format and verification rules.

## Local evaluation

The initial experiment recovered 66 of 78 staged copies and selected no
region in 104 controls from 26 photo groups. The separate panel used 38
attributed Wikimedia Commons photos from the existing geometric-copy archive,
four native crop sizes, new placements, rectangular patches, and clockwise
rotation in the feathered case. The new detector uses the same measurement
settings as the initial experiment. The archive records CC BY or CC BY-SA
terms for every source. Ordinary controls include the unchanged crop,
global tone adjustment, local blur, straight copy and reflected copy.
[Source archive, terms and attribution](image-affine-native-scale-2026-11-11.md)

A useful localization reaches 0.50 intersection over union on both
of the known source areas. The comparison runs the prior
same-orientation and geometric copy evidence paths on their supported
sizes. The tested files are staged PNG deliveries from previously
collected photos. These counts measure pixel correspondence on this
panel. They do not establish prevalence or accuracy for arbitrary
images or generative edits.

| Local held cases | Useful locations before | Useful locations with rotated-copy review |
| --- | ---: | ---: |
| Quarter turn, 24 × 28 px | 0/38 | 36/38 |
| Half turn, 36 × 28 px | 0/38 | 37/38 |
| Feathered clockwise quarter turn, 44 × 42 px | 0/38 | 35/38 |
| New highlights on 190 matched controls | | 0 |

## Local app verification

Four full analyses used simulated USD 9.99 checkout. No analysis job
started before payment. One quarter-turn input ran with the option off and
on, producing the same overall assessment both times. A half-turn input
produced a second exact pair and the ordinary control produced no pair.
The API and JSON agreed on the native boxes and measured alignment.
Both PDFs show the two unrotated crops from the image and describe the
rotation. Both signed packages recomputed exact pixel agreement and the
surrounding differences from the submitted source.
[Local app receipt](evaluation-panels/image-exact-turned-copy-app-2026-11-15.json)

## Replay

The independent reader needs only standard-library Python to recount
the position overlaps and confirm the output orientation and pixel
counts. The native replay also recreates every delivery, checks the
original source and decoded RGB digests, and recomputes each
measurement directly from the crops. The public receipt contains
numeric positions, measurement values and SHA-256 commitments.
[Held results](evaluation-panels/image-exact-turned-copy-held-2026-11-15.json)

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
receipt=docs/evaluation-panels/image-exact-turned-copy-held-2026-11-15.json
python3 -B -S -m scripts.image_exact_turned_copy_transfer "$receipt"
backend/.venv/bin/python -B -m scripts.image_exact_turned_copy_transfer \
  "$run/held.json" --archive "$archive"
cmp "$receipt" "$run/held.json"
```

The local app proof uses the same archived source assets and a fresh
test database.

```sh
backend/.venv/bin/python -B -m scripts.image_exact_turned_copy_app \
  "$run/app" "$receipt" "$archive"
```
