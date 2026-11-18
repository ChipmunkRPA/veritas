# Two exact copy pairs in one image

Chipmunk Verify can show two separate pairs of matching areas in one
lossless image. Each pair keeps two unaltered source crops, their native
coordinates and the alignment that makes their RGB pixels match. The ordinary
view links each crop to its position in the original. The PDF and JSON
retain both comparisons, including the number of matching pixels and the
differences around the aligned crops. Signed source replay verifies each
pair directly against the submitted image.

The check compares upright, static RGB PNGs with both sides between 384 and
768 pixels, or opaque, simple lossless WebP stills with both sides between
512 and 768 pixels. It can combine same-orientation matches, quarter and
half turns, and horizontal and vertical flips. It reuses the existing
requirements for texture, exact interiors of 16–56 pixels per crop side,
and differences along at least three sides of the three-pixel surroundings.

The bounded search must find exactly two qualifying pairs with all four
endpoints separated by at least 16 pixels across pairs. It withholds the
two-pair result on shared or nearby endpoints, a third qualifying match or an exceeded
search budget. Existing comparisons remain available when it returns no
two-pair result. Two verified exact pairs can take the place of a single
previous copy match. A trusted AI declaration still skips the image copy
checks. Earlier signed reports retain their original evidence version.
Other duplicated or edited areas can fall outside the supported pixel
conditions or search budget.

Edited copies, deliberate graphic symmetry and repeated patterns can all
produce exact matches. The result establishes where pixels agree. The
sequence of edits and AI involvement remain undetermined. These comparisons
carry zero weight in the automated authorship assessment. The work is
motivated by studies that evaluate several local edits separately. It uses
no third-party detector code or weights.
[CVPR Findings study on multiple edits](https://openaccess.thecvf.com/content/CVPR2026F/html/Zhu_Revisiting_Image_Manipulation_Localization_under_Realistic_Manipulation_Scenarios_CVPRF_2026_paper.html)

`VERITAS_IMAGE_DISJOINT_EXACT_COPY_REVIEW=true` enables the optional local
development comparison. The setting defaults to false in the shared code.

## Held comparison

A separate local run froze crop positions and transforms across 18
camera-photo groups from the CC0 RAW sample archive. Separate ordinary
controls included intact architecture with symmetry and repeated structure.
Each positive applies two edits with known source and destination boxes.
The raw denominators include low-texture edits and edits with too little
surrounding contrast. [RAW sample archive and per-file licenses](https://raw.pixls.us/)
· [Pinned camera roster](evaluation-panels/image-small-affine-jpeg-cameras-2026-11-16.json)
· [Architecture source credits](evaluation-panels/image-affine-native-scale-2026-11-11.json)

| Authored pairs | Previous app checks located both | With the exact two-pair check |
| --- | ---: | ---: |
| Two translations | 0/18 | 11/18 |
| Two reflections on the same axis | 0/18 | 12/18 |
| Two reflections on different axes | 0/18 | 11/18 |
| A translation and a reflection | 0/18 | 11/18 |
| Two rotations | 0/18 | 12/18 |
| All staged two-pair edits | 0/90 | 57/90 |

All 114 returned pairs matched the authored boxes. The previous checks
found 11 of those pairs. The new comparison added 103 correctly located
pairs and preserved every previously localized true pair. The new check showed no
pairs on 27 ordinary photos, six whole-image symmetry controls, 18 shared
endpoints or 18 nearby endpoints. It withheld all 12 three-edit files whose
three individual pairs met the pixel requirements. In three of the other
six deliberately three-edited files it found exactly the two eligible
authored pairs. This misses the panel's strict condition of returning no
pairs on any three-edit file, and shows that the comparison cannot count
all edits. The local evidence therefore reports two qualifying pairs and
states that other areas can fall outside the search.

A 512 × 576 transfer used the same source groups and enabled the existing
geometric comparison. Full localization rose from 1/90 to 51/90, adding
85 correct individual pairs with no loss of previous true pairs or new
finding on the 27 ordinary photos. Six of the 18 deliberately three-edited
files returned exact two-pair subsets. All 10 files in which all three
pairs qualified were withheld. Both sizes use the same source groups.
They provide two delivery checks with one set of independent originals.

The [main evaluation receipt](evaluation-panels/image-two-exact-copy-pairs-2026-11-18.json.gz)
and the [512 × 576 transfer](evaluation-panels/image-two-exact-copy-pairs-512x576-2026-11-18.json.gz)
contain source and delivery hashes, the frozen edit rectangles, and
recorded candidate and previous checks. It contains no photographs or
crops. The standard-library recount compares both recorded boxes and
transforms against the authored geometry.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_image_two_exact_pairs \
  docs/evaluation-panels/image-two-exact-copy-pairs-2026-11-18.json.gz
```

Original-source replay uses the downloaded signed package, the trusted
signing key ID, and the original upload. It checks the coordinates of all
four crops and each pair's pixel count, transform, and surrounding differences.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m app.cli verify report.zip \
  --expect-key-id KEY_ID --source-image original.png
```

## Local application verification

Two authored lossless fixtures combined one same-orientation match and one
vertical flip. With the option off, each showed one pair. With the option
on, each showed two exact pairs. The ordinary browser view linked all four
crop locations per format. Its PDF, JSON and signed replay captured the
same source coordinates and measurements. The test completed four simulated
paid analyses and started zero jobs before payment. The overall assessment
matched with the option off and on. Earlier evidence remained readable.
[Local browser and export receipt](evaluation-panels/image-two-exact-copy-pairs-app-2026-11-18.json)
