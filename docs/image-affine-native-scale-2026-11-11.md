# Verify geometric copy evidence on larger images

The geometric copy check now supports a long side of 1,025–2,048 pixels and a short side of at least 768 pixels in local development. It searches for candidate pairs in a reduced image with a 1,024-pixel long side, then fits the same candidate coordinates to the oriented source pixel grid. Each shown pair has two exact source crops and binary masks. The interface and PDF state the working size, and the JSON records it with the exact source geometry. The check carries zero AI authorship confidence weight. [Earlier geometric comparison](image-affine-copy-transfer-2026-11-11.md)

The proposal transform is conjugated by the sample-center coordinate mapping for the exact two image sizes. The lifted matrix is rounded to the saved six-decimal precision before scoring. The existing native luma support check tests the aligned original pixels, retains a connected area near the seed, and maps it to the second area. Cross-scale correspondence research motivates separating proposals from source-pixel evidence. This implementation uses no downloaded model, code, or weights from the papers. [Cross-scale correspondence research](https://arxiv.org/abs/2404.17310)

## Local comparison

A frozen Wikimedia Commons roster selected 38 photos with distinct credited contributors from six architecture, cityscape, and landscape categories. All used CC BY or CC BY-SA. The first intake obtained two publisher originals. Its thumbnail host check incorrectly rejected the CDN returned by the metadata service, and the original host began returning HTTP 429. The corrected intake retained both verified originals and obtained all 36 remaining publisher thumbnails from their already specified URLs. No source was replaced. For the run, both originals were resized to a 1,920-pixel long side and all deliveries were measured at that size. The ordinary controls are unmodified evaluation photos or known ordinary transforms. Camera provenance and alias independence are unverified. [Source credits, hashes, and counts](evaluation-panels/image-affine-native-scale-2026-11-11.json)

The 456 views include a rotated and resized sharp copy, a feathered copy, the unmodified input, a global contrast change, a local tone adjustment, and a resize round trip for every source in PNG and JPEG75. A useful location requires intersection over union of at least 0.30 and recall of at least 0.50 for both authored regions. The original-size affine check previously left every view unassessed. The comparison does not estimate how the wider method compares to the separate reflection check.

| Delivery | Sharp localized | Feathered localized | Ordinary views flagged | Ordinary source groups flagged |
| --- | ---: | ---: | ---: | ---: |
| PNG | 32/38 | 25/38 | 4/152 | 1/38 |
| JPEG75 | 30/38 | 23/38 | 3/152 | 1/38 |

Both recipes assessed every view and passed the frozen coverage and false-flag criteria. Ordinary flags came from one source group, including its unmodified view in both formats. The selected recipe preserves the prior source-resolution smoothing. Scaling that smoothing to match the reduced image lost five useful localizations and removed no ordinary flags. These figures measure ordinary copying, including feathered blending. Neither the copied regions nor the flagged ordinary views establish AI involvement.

The local CPU run took 227 seconds. The separate standard-library reviewer checked all 38 acquired source files, 456 view records, and 564 mask-side overlap records. The browser, PDF, signed package, and original-file replay use the same saved matrices and native masks. Older image evidence remains readable.

## Reproduce

The public aggregate supplies the exact roster and intake. Copy each bound source to the corresponding `name` from the acquisition list and check its SHA-256. Use `thumb_url` or `original_url` from the corresponding roster slot according to the recorded delivery. Recreate the two JSON inputs with sorted keys, two-space indentation and a final newline.

```sh
python - path/to/image-affine-native-scale-2026-11-11.json "$run_dir" <<'PY'
import json
import sys
from pathlib import Path

public = json.loads(Path(sys.argv[1]).read_bytes())
destination = Path(sys.argv[2])
destination.mkdir(parents=True, exist_ok=True)
for key, name in (("source_roster", "plan.json"), ("source_intake", "acquisition.json")):
    (destination / name).write_text(json.dumps(public[key], indent=2, sort_keys=True) + "\n")
PY
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.image_affine_native_scale freeze "$run_dir"
backend/.venv/bin/python -B -m scripts.image_affine_native_scale run "$run_dir"
backend/.venv/bin/python -B -m scripts.review_image_affine_native_scale "$run_dir"
```

The raw photos and derived crops stay in the evaluator's workspace. Keep the listed attribution and source terms with any redistributed media.
