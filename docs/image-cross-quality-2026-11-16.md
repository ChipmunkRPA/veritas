# Locating small edits across JPEG quality settings

Three dense image checks found no additional correctly placed highlights on
720 held JPEG deliveries. They used exact luminance coefficients and decoded
color residuals. Those three fits remain in evaluation. The separate
JPEG90 repeat check below is available in local development.

## Method and source groups

The test asks whether training on two JPEG saves of the same edited photo
helps the check locate the edit across compression settings. Recent research
describes cross-domain feature alignment and the effect of distant image
content on inpainting detection. This implementation tests local measurements
against exact edit coordinates and does its own fitting. No external code or
weights were added. [Cross-domain study](https://www.mdpi.com/2227-7390/14/22/4188)
· [Inpainting study](https://arxiv.org/html/2602.00192v2)

Use three measurements of 20 luminance coefficient positions and three bands
of decoded color residuals on each 16 × 16 cell. Compare each cell with
its neighborhood and the surrounding ring. Fit a dense linear response
directly on edited cells, rival regions, and exact locations on matched
ordinary photos. The three fits use separate slopes at each quality, a
shared slope, and a shared slope that penalizes unequal responses at the
same location across the two saves. Each method selects one connected
component. A useful highlight needs at least 30% intersection over union
and 50% coverage of the actual pixels changed before encoding.

Seventy-two previously studied TGIF and TGIF2 source photos supply fitting
data. A further 24 set review points and choose the method. They are
separate from the 36 retained BG-REAL LaMa transfers and from the 24 newly
acquired BG-REAL photos. The new set contains 12 object removals and 12
object insertions from different generators. The source preparation
excludes previously recorded photo identifiers, original digests and RGB
digests. The retained transfer photos had already appeared under other
methods. Earlier exposure in the training of pretrained models is unknown.
The BG-REAL source receipt records original photo attribution, delivery
hashes and the pinned dataset revision. [TGIF and TGIF2 release](https://github.com/IDLabMedia/tgif-dataset/tree/fc5c557bd1a2d49a5dabc3676a54cfc44798f4ea)
· [BG-REAL and photo terms](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)
· [New source roster](evaluation-panels/image-cross-quality-sources-2026-11-16.json)

Each source supplies a sharp and a softly blended edit, an untouched
photo, and ordinary local blur, tone correction and copying controls.
All six files were encoded at quality 75 and 90 with 4:2:0 sampling.
The two saves and controls at the same location share one source group.

## Held results

The separate fits were selected using calibration results before measuring
the holdout. Their wider setting allowed two calibration source groups with
additional ordinary flags at each quality. The stricter setting allowed
zero. A new highlight counts as an addition only when the existing checks
show no highlight. The app comparison includes the aligned JPEG compression
check on all 720 deliveries and the regional review where it currently
applies, on 342 quality-75 deliveries.

| Panel and JPEG quality | Current useful | Current ordinary flags | Extra ordinary flags with wider fits, separate / shared / matched saves |
| --- | ---: | ---: | ---: |
| Fresh · 75 | 11/48 | 7/96 | 69 / 25 / 25 |
| Fresh · 90 | 0/48 | 0/96 | 16 / 8 / 8 |
| Retained LaMa · 75 | 0/72 | 15/144 | 102 / 38 / 55 |
| Retained LaMa · 90 | 0/72 | 0/144 | 32 / 24 / 20 |

At either review point, every fit added zero useful locations. The
matched-save fit reduced ordinary highlights relative to the separate
fits at the wider point. Even its stricter version added two ordinary
highlights on the retained quality-75 deliveries. A more permissive
review point on this head would trade additional false alarms for regions
that usually fall outside the edit. These results describe the tested
delivery pipeline and source groups. [Per-delivery receipts](evaluation-panels/image-cross-quality-2026-11-16.json.gz)

## Exact input and overlap replay

A separate local pass checked 1,872 encoded file hashes, 936 exact truth
masks, and 2,160 native crop intersections. The closest of 3,168
fresh-to-retained photo comparisons differed in 182 of 512 thumbnail
gradient bits. A restore into an empty directory regenerated all 288 new
JPEGs from pinned public inputs and reproduced the exact input manifest.
[Readback totals](evaluation-panels/image-cross-quality-audit-2026-11-16.json)

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
published=docs/evaluation-panels
receipt="$published/image-cross-quality-2026-11-16.json.gz"
python3 -B -S -m scripts.image_cross_quality_fit verify "$receipt"
backend/.venv/bin/python -B -m scripts.image_cross_quality_sources restore \
  "$published/image-cross-quality-sources-2026-11-16.json" "$run/fresh"
```

To regenerate the retained files and their current comparison, follow
the [earlier delivery preparation](image-region-ranking-2026-11-16.md#replay-and-compare-another-check).
Place the retained manifest below `$retained/deliveries` and its
current-check readings at `$retained/reference.jsonl`. Run the reference
tool on the new deliveries before assessment. The native audit reopens
both media roots and verifies the frozen regions, selected areas and
changed-pixel intersections.

```sh
runner=backend/.venv/bin/python
"$runner" -B -m scripts.image_region_rank_reference "$run/fresh" "$run/fresh-reference.jsonl"
"$runner" -B -m scripts.image_cross_quality_fit fit "$retained" "$run/model-freeze.json"
"$runner" -B -m scripts.image_cross_quality_fit assess "$run/model-freeze.json" \
  "$retained" "$run/fresh" "$run/fresh-reference.jsonl" "$run/results.json.gz"
"$runner" -B -m scripts.audit_image_cross_quality "$retained/deliveries" "$run/fresh" \
  "$run/results.json.gz" "$run/native-audit.json"
```

## Compare the existing regional check on a JPEG90 upload

The previous regional study compared JPEG90 variants with an optional
regional pass on JPEG75 originals. This replay compares the JPEG90
variants with what the app actually runs on the same JPEG90 bytes.
It applies the earlier region and agreement rules without new fitting.
[Earlier agreement study](image-regional-jpeg90-quality-agreement-2026-11-13.md)

Measure the native image and a same-size JPEG75 re-encoding. Both passes
use the original JPEG90's quality exclusions. The primary comparison
keeps the exact native component if at least half its measured cells
also appear in the second pass. It also records the native and re-encoded
regions separately. This is a sensitivity check with one image and the
same weights, so an agreeing region carries zero AI-authorship weight.

| JPEG90 panel | Correct regions, current / native / re-encoded / agreement | Ordinary flags, current / native / re-encoded / agreement |
| --- | --- | --- |
| 24 fresh sources | 0 / 11 / 11 / 11 of 48 | 0 / 24 / 3 / 0 of 96 |
| 36 retained LaMa sources | 0 / 0 / 0 / 0 of 72 | 0 / 29 / 15 / 3 of 144 |

The agreement rule finds 11 of 24 insertion variants from six new source
photos, with one additional misplaced highlight. It finds zero of the
24 new removal variants. It assesses all 144 fresh JPEG90 deliveries
and 198 of 216 retained deliveries. The remaining 18 fall below the
minimum image size. There are two retained ordinary source groups with
flags. These results support a limited local inspection option for the
tested JPEG profile. Reliable coverage of object removals and other
encoding profiles remains open. [JPEG90 results](evaluation-panels/image-jpeg90-agreement-2026-11-16.json.gz)

### See the check in Chipmunk Verify

Enable `VERITAS_IMAGE_REGION_JPEG90_REVIEW=true` and
`VERITAS_IMAGE_REGION_JPEG_REVIEW=true` in development and run Full
analysis. The option covers upright 8-bit RGB baseline 4:2:0 JPEGs with
the standard quality-90 tables, a 768-pixel long side and a short side of
at least 512 pixels. The existing quality-75 path is separate.

The evidence card shows the original crop and the exact retained area.
The browser and Standard PDF say why the area was selected and state the
limits. JSON includes source bounds, overlap counts and the second view's
hash. With the original and the same JPEG encoder, the signed package
verifier checks the crop pixels, rebuilds the second JPEG view and
recounts overlap from the stored masks. That verification uses no model
inference.

The simulated USD 9.99 checkout replay queued no analysis before a paid
event. In the tested edit it kept the region at **x 377, y 456**, with
size **128 × 76 pixels**. **8,658 of 8,668** measured cells overlapped.
The second native region was withheld. The browser, PDF and JSON showed
the same coordinates and measurements, with model names omitted.
[App replay receipt](evaluation-panels/image-jpeg90-agreement-app-2026-11-16.json)

```sh
export PYTHONPATH=backend
published=docs/evaluation-panels
python3 -B -S -m scripts.image_jpeg90_agreement_holdout verify "$published/image-jpeg90-agreement-2026-11-16.json.gz"
backend/.venv/bin/python -B -m scripts.image_jpeg90_agreement_holdout score \
  "$retained" "$run/fresh" "$run/fresh-reference.jsonl" "$run/agreement.json.gz"
npm --prefix frontend run build
export CHIPMUNK_REGIONAL_JPEG90_FLOW_INPUT="$run/fresh/fresh-018/sharp-90.jpg"
export VERITAS_DEVICE=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
backend/.venv/bin/pytest backend/tests/test_native_regional_jpeg_paid.py -o addopts='' -m slow -k jpeg90-repeat
```
