# Local opacity changes in an earlier-copy PNG comparison

Open **Compare with an earlier copy** in a completed local image analysis.
The comparison now detects a cutout, reveal or partial transparency edit
between suitably matched PNGs, even when the stored RGB values stay the
same. It shows the opacity pixel count, the selected region mask, and
earlier and submitted crops composited on white in the browser, PDF and
JSON export. The original files and exact decoded RGBA values have
separate SHA-256 commitments. The comparison carries zero weight in AI
confidence. [PNG opacity representation](https://www.w3.org/TR/png-3/#6AlphaRepresentation)

The route covers single-frame 8-bit RGB and RGBA PNGs under four million
pixels with matching oriented dimensions and color metadata. At least
12 spread-out, textured and fully opaque areas must match exactly.
Matching visible samples must cover at least 65% of the image, and the
bounding box of differences can cover up to 40% of it. The mask includes
opacity changes and changed RGB samples where either pixel is visible.
Pixels that are fully transparent in both copies contribute no visible
color changes. The route displays up to five components under the
existing minimum-size rule. Fully opaque pairs continue through the
existing image comparison. A large or poorly aligned transparency
change can leave the pair unassessed.

## Local comparison

The test used eight pinned, credited Wikimedia photographs previously
seen by the project. The PNG pairs are newly authored. Each photo supplied
a local cutout, partial-opacity change and reveal at a known 33 × 29
pixel location. Controls covered opaque re-encoding, hidden RGB changes,
whole-image opacity and rotation. The table counts exact locations for
the authored local edits and any region for the controls. Media stays
local. [Public source delivery, license and attribution roster](evaluation-panels/image-jpeg-public-replay-2026-11-12.json)

| Pair condition | Existing RGB route | Opacity-aware comparison |
| --- | ---: | ---: |
| Local cutout, exact location | 0/8 | 7/8 |
| Partial opacity, exact location | 0/8 | 7/8 |
| Reveal, exact location | 0/8 | 7/8 |
| Fully opaque re-encode, highlighted | 0/8 | 0/8 |
| Hidden RGB only, highlighted | 7/8 | 0/8 |
| Whole-image opacity, unassessed | 0/8 | 8/8 |
| Rotated transparent PNG, unassessed | 8/8 | 8/8 |

The same photo lacked sufficient textured witnesses in the three local
edits and the hidden-color case. There were seven comparable hidden-color
pairs, each with zero visible differences. The counts measure PNG
comparison on the eight photos. An earlier copy is required for this
check. [Aggregate receipt](evaluation-panels/image-reference-png-opacity-2026-11-14.json)

The authenticated local endpoint returned the exact 483-pixel mask for
a separate partial-opacity fixture. Both crops and the overlay matched
the PDF's embedded images. The original-source reviewer recomputed
the comparison and rejected a modified opacity count.

## Reproduce

Prepare the public photos with the [pinned acquisition](image-jpeg-public-replay-2026-11-12.md#run-from-a-clean-checkout).
The comparison runner accepts that public delivery directory.

```sh
PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  backend/.venv/bin/python -B -m scripts.image_png_opacity_transfer "$replay_data"
```

The earlier-copy [export reviewer](../backend/scripts/review_image_reference_export.py)
checks both source files, the decoded PNG comparison, the mask and crop
pixels from the browser's JSON download.
