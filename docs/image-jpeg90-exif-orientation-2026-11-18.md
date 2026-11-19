# Regional JPEG evidence on rotated and reflected uploads

The regional image check can now inspect supported quality-90 JPEGs
that carry a saved rotation or reflection. It runs on the decoded
pixels in their stored order. The JPEG75 repeat uses the same stored
pixels as before. Only after measurement does the app rotate or
reflect the quality mask, candidate masks and bounding boxes to match
the displayed image. Original crops use those displayed coordinates.
[Existing quality-90 test and scope](image-cross-quality-2026-11-16.md#progressive-jpeg-transport)

The new option covers single-frame RGB baseline and supported progressive
4:2:0 JPEGs with the standard quality-90 tables, a 768-pixel long side
and a short side of at least 512 pixels. It accepts the seven standard
EXIF display transforms. The existing whole-image rule, region ranking
and repeat requirement apply without a new threshold. The finding has
zero weight in the overall AI assessment. JPEG compression can obscure
small or softly blended edits. [CIPA EXIF display orientation](https://www.cipa.jp/e/std/std-sec.html)

## Exact transport

The local test added an EXIF orientation tag to each of the 360 already
measured JPEGs, cycling through the seven display transforms. It left the
encoded JPEG picture unchanged. Both Pillow and a separate libjpeg
decoder reproduced the stored color pixels on all 360 pairs. All 342
eligible pairs had identical input tensors, quality masks and JPEG75
repeat bytes. Eighteen files fell below the minimum dimensions.
For a known edit, an unchanged file and an ordinary blur alert, we also
ran every transform on both baseline and progressive JPEG. All 42 model
comparisons reproduced the same selection, repeat overlaps, quality
exclusions and relative rankings on their corresponding display grids.

The inherited eligible results cover 11 useful locations on 48 staged
edits in the fresh comparison bank, with zero highlights on 96 ordinary
files. The retained removal bank still has zero useful locations on 72
staged edits and three ordinary highlights. These checks test exact
transport of an existing measurement. Accuracy on new JPEG profiles
and different types of localized edits remains unmeasured.
[Previous case labels and numeric decisions](evaluation-panels/image-jpeg90-agreement-2026-11-16.json.gz)

## Local application

Two simulated USD 9.99 checkouts verified payment before inference
and reproduced the browser, PDF, JSON and signed evidence packages
against the uploaded files. The 90-degree clockwise tag moved the
original crop from x 377, y 456 at 128 × 76 pixels to x 44, y 377 at
76 × 128 pixels. The repeat still shared 8,658 of 8,668 selected cells.
[Local native and application receipt](evaluation-panels/image-jpeg90-exif-orientation-2026-11-18.json)

Enable the stored-pixel route in a development backend with

```sh
VERITAS_IMAGE_REGION_JPEG_REVIEW=true
VERITAS_IMAGE_REGION_JPEG90_REVIEW=true
VERITAS_IMAGE_REGION_JPEG90_ORIENTATION_REVIEW=true
```

The opt-in applies to new analyses. Earlier upright JPEG reports keep
the re-encoding and pixel-coordinate rules they recorded when signed.

The public receipt can be checked against the published baseline case
records. The original assets are needed to repeat the two-decoder
verification and model inference.

```sh
python3 -I -S - <<'PY'
import gzip, hashlib, json
from collections import Counter
from pathlib import Path

root = Path("docs/evaluation-panels")
old = (root / "image-jpeg90-agreement-2026-11-16.json.gz").read_bytes()
new = json.loads((root / "image-jpeg90-exif-orientation-2026-11-18.json").read_bytes())
assert hashlib.sha256(old).hexdigest() == new["commitments"]["previous_case_records_sha256"]
counts = {}
for row in json.loads(gzip.decompress(old))["rows"]:
    tally = counts.setdefault(row["split"], Counter(dict.fromkeys(new["native_transport"]["panel_counts"][row["split"]], 0)))
    tally["stored_rgb_pairs"] += 1
    if row["agreement"] is not None:
        tally["eligible"] += 1
        hit, selected, truth = [row["agreement"][k] for k in ("intersection", "selected", "truth")]
        tally["inherited_useful"] += bool(truth and 2 * hit >= truth and 10 * hit >= 3 * (selected + truth - hit))
        tally["inherited_ordinary_flags"] += bool(not truth and selected)
assert counts == new["native_transport"]["panel_counts"]
print("Published eligibility, location and ordinary-control counts agree.")
PY
```

To repeat the local checkout from the retained `fresh-018/sharp-90.jpg`
fixture described in the prior study, point the test at that file. The test
checks its digest and adds the EXIF tag locally.

```sh
export CHIPMUNK_REGIONAL_JPEG90_ORIENTED_FLOW_INPUT="$run/fresh/fresh-018/sharp-90.jpg"
backend/.venv/bin/python -B -m pytest -o addopts='' \
  backend/tests/test_native_regional_jpeg_paid.py -m slow -k jpeg90-oriented-repeat
```
