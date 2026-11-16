# Compare a lossy WebP with an earlier JPEG

The earlier-copy tool maps local brightness differences when an upright
JPEG and an opaque lossy WebP share the same framing. It links each marked
region to the decoded crops of both supplied files and highlights the
selected pixels on the submitted crop. The PDF and JSON keep those
coordinates, crops and source hashes. The comparison requires an earlier
copy from the user. Ordinary editing can produce the same differences,
so the result contributes zero weight to the overall AI assessment.

## Local comparison

The panel contains 48 source photos and their published local edits, eight
per editor in each of two source-disjoint retained TGIF banks. The project
has used these banks in earlier studies. The input builder verifies the
published RGB pairs and masks and retains native dimensions of 512 to
1,024 pixels per side. Each photo is delivered with three encoding recipes.
See the [pinned publisher materials and source terms](https://github.com/IDLabMedia/tgif-dataset/tree/fc5c557bd1a2d49a5dabc3676a54cfc44798f4ea).

The earlier copies use baseline JPEG at quality 75 with 4:2:0 sampling,
progressive JPEG at quality 82 with 4:2:0, and baseline JPEG at quality 94
with 4:4:4. The respective submitted WebPs use qualities 65, 82 and 92.
For each pair, the existing framing check requires near-zero translation
and agreement across textured areas. The brightness check then smooths
both images, accounts for a global offset, and selects surviving local
differences on the submitted pixel grid. A useful result covers at least
half the known edit and reaches 0.30 intersection over union.
[WebP bitstream container specification](https://developers.google.com/speed/webp/docs/riff_container)

| Submitted pair | Deliveries | Compared | Any difference marked | Useful location |
| --- | ---: | ---: | ---: | ---: |
| Published local edit | 144 | 120 | 120 | 111 |
| Original photo encoded in both formats | 144 | 120 | 0 | — |
| Earlier JPEG decoded and saved as WebP | 144 | 120 | 0 | — |
| Overall brightness offset | 144 | 120 | 7 | — |
| Conventional local tone change | 144 | 120 | 106 | 53 |
| Conventional local blur | 144 | 120 | 0 | 0 |
| Four-pixel translation | 144 | 0 | 0 | — |

The 111 useful results come from 38 of 48 source photos. Thirty-six photos
give a useful location in all three encoding recipes. The 33 remaining edit
deliveries include 24 unassessed pairs and nine measured locations below
the overlap requirement. The three encodes of each photo share the same
underlying pixels. The extra color measurement found no additional useful
published edit and added one ordinary local tone highlight. The JPEG
route uses the brightness measurement.
[Frozen comparison receipt](evaluation-panels/image-jpeg-webp-comparison-2026-11-16.json.gz)

The local toggle is `VERITAS_IMAGE_REFERENCE_LOSSY_WEBP_EARLIER_JPEG=true`
alongside the existing earlier-copy and lossy WebP development settings.
It is off in the default configuration. The route covers upright
single-image RGB JPEG without a separate color profile, with baseline or
progressive coding and 4:2:0 or 4:4:4 sampling. JPEG and WebP dimensions
must match. The production comparison reproduced all 1,008 frozen cases,
including the hashes of the selected masks. The earlier lossless-reference
route declined all 1,008 of these JPEG reference pairs.

## Local app

Three simulated USD 9.99 checkouts replayed an edited WebP, a WebP
re-encode of the earlier JPEG, and a shifted WebP. No analysis queued
before payment. With the option enabled, the edited file showed one
region covering 14,088 pixels, 13,812 of them inside the published edit.
The re-encode showed zero regions. The shifted pair was unassessed.
The normal view rendered both source crops and the overlay. The browser
downloads matched the API response. The PDF contains the same exhibit
pixels, and the JSON verifier rebuilds the source crops and mask from
the original files. The original analysis result was unchanged.
[Prepaid browser, PDF and JSON receipt](evaluation-panels/image-jpeg-webp-comparison-app-2026-11-16.json)

## Single-image alternatives

The current PNG regional check, applied to the same pixels encoded as
lossy WebP, supplied useful edit locations on 5 of 24 WebP68 deliveries
and 9 of 24 WebP90 deliveries. It marked 29 of 72 and 27 of 72 ordinary
deliveries, respectively. The PNG comparison had 15 useful locations and
21 ordinary alerts. Requiring the same regions to survive a second JPEG
encode reduced ordinary flags to 7 and 9, and retained only three useful
locations at either WebP quality. The three delivery variants share the
same 24 source photos.
[Regional transfer receipt](evaluation-panels/image-regional-lossy-webp-2026-11-16.json.gz)

A separate experiment compared local Fourier phase, power and their
combination. Each component fit used one source bank, a second bank
selected the operating points and a third supplied 24 held photo groups.
The phase-only fit gave one useful PNG location, one misplaced PNG flag
and zero ordinary PNG flags. None of the three fits located a held
WebP90 edit. The combined fit gave one ordinary WebP90 alert. The search
used project-written color-channel features and brought in no third-party
code or weights. These single-image candidates remain in evaluation.
[Phase and amplitude research](https://arxiv.org/html/2606.04545v1#S4.SS1)
· [Frozen phase comparison](evaluation-panels/image-local-phase-transfer-2026-11-16.json.gz)

## Replay

The numeric readers require standard-library Python. The paired native
replay uses the local source banks and verifies delivered and selected
pixels against the frozen receipt. The original media can be reacquired
from the publisher by the source IDs and editor roles in the receipts.
The local bank inputs follow the manifest format generated by
`backend/scripts/prepare_tgif_localization.py`. Media and weights stay
outside the published receipts.
[Source preparation](image-localizer-upgrade.md)

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
published=docs/evaluation-panels
paired="$published/image-jpeg-webp-comparison-2026-11-16.json.gz"
python3 -B -S -m scripts.image_reference_webp_earlier_jpeg review "$paired"
python3 -B -S -m scripts.image_regional_lossy_webp_transfer review \
  "$published/image-regional-lossy-webp-2026-11-16.json.gz"
python3 -B -S -m scripts.image_local_phase_transfer review \
  "$published/image-local-phase-transfer-2026-11-16.json.gz"
backend/.venv/bin/python -B -m scripts.image_reference_webp_earlier_jpeg replay \
  "$paired" "$sources"
backend/.venv/bin/python -B -m scripts.image_reference_webp_earlier_jpeg_app \
  "$paired" "$sources" "$run/prepaid-app" --browser
```
