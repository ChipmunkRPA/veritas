# Independent review of the reserved JPEG aggregate

Run the aggregate review after the v189 trial finishes, the terminal seal
succeeds and the exact v189 aggregate is published. It verifies the previously
published v189 freeze and its v42/v46 model, readout and runtime commitments.
It also checks every frozen code or protocol file by SHA-256. The reviewer
uses the same reserved 59 photos in 48 source groups at JPEG75 and JPEG90.
[v189 protocol](image-jpeg-diagnostic-v189-protocol.md) ·
[Published freeze](evaluation-panels/image-jpeg-diagnostic-freeze-v189.json)

## Evidence and arithmetic

The published aggregate and the private trial's `summary.json` must match
the supplied aggregate digest. The reviewer requires the successful seal
before opening any measurement journal. It verifies preparation and all
24 paired phases, including 48 producer and 24 readout profiles and their
resource-journal digests. It verifies each completion record against its
exact phase and freeze and checks its stated measurement-journal digest.
The aggregate itself commits the 73 preparation, producer and readout
profiles. The completion and measurement-journal digests come from their
private receipts and are retained in the new review receipt. The aggregate
does not directly commit those per-photo journals.

For each of the 590 measurements the reviewer checks the view, quality,
source group, geometry and footprint against the frozen input manifest.
It checks both producer rows and their input bindings. The current gate is
recomputed from the recorded scalar. Each method's contingency counts
independently determine all pixel ratios and the exact useful-localization
criteria. The recorded shared counts must fit the possible intersection
bounds inside and outside the truth mask. Tone, blur and copy-move use zero
AI truth even when the detector overlaps their separately recorded ordinary
footprints. Empty ordinary footprints stay in the control denominator and
have no footprint-specific IoU or recall.

Group means use exact rational values. The paired bootstrap draws the
published number of groups with replacement from the same fixed PCG64
seed, then converts each draw to a vector of group multiplicities. A
weighted sum and explicit sorted-percentile interpolation reproduce the
interval endpoints independently of the production scoring implementation.
Comparison requires all result fields, exact counts and a maximum absolute
tolerance of `2e-12` for floating-point values. Intervals are conditional
and pointwise. The two qualities reuse the same 48 source groups and do not
represent two independent cohorts.

This review verifies arithmetic and the binding of recorded measurements.
It opens no raw JPEGs, reference masks, response maps or checkpoints.
It cannot independently verify the pixels of a localization or reconstruct
the candidate gate from its raw response. It does not run models, estimate
authorship confidence or support a deployment decision by itself.

## Commands

From the project root, verify the public identities without opening the trial.

```sh
PYTHONPATH=backend backend/.venv/bin/python -m scripts.review_image_jpeg_aggregate_v207 check-public
```

After publication, set `published` to the exact checked-out aggregate path
and `published_sha` to its independently verified SHA-256. Use a new private
receipt path. This command requires NumPy and ordinary filesystem reads. It
does not require Metal, a network connection or model dependencies.

```sh
trial="$PWD/backend/data/independent-evaluation/image-jpeg-diagnostic-v189"
published="$PWD/docs/evaluation-panels/<published-v189-aggregate>.json"
published_sha="<verified SHA-256 of the published aggregate>"
receipt="$PWD/backend/data/independent-evaluation/image-jpeg-aggregate-review-v207.json"
PYTHONPATH=backend backend/.venv/bin/python -m scripts.review_image_jpeg_aggregate_v207 review "$trial" "$published" "$published_sha" "$receipt"
```

Synthetic local verification covers source-group pairing, unequal group
sizes, integer threshold ties, wrong false area, impossible intersections,
journal tampering and exact public identities.

```sh
cd backend
.venv/bin/python -m pytest -q -o addopts='' -p no:cacheprovider tests/test_review_image_jpeg_aggregate_v207.py
```
