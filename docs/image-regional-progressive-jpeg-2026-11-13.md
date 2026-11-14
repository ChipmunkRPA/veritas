# Regional image evidence for progressive JPEG

The local Full and Deep image check now accepts single-frame progressive
JPEGs within its existing quality-75, 4:2:0, upright RGB profile. The long
side must be 768 pixels and the short side at least 512 pixels. The source
parser checks the scan boundaries and requires fixed quantization tables.
The existing development option `VERITAS_IMAGE_REGION_JPEG_REVIEW` controls
the route. It continues to carry zero weight in overall AI confidence.

## Local verification

Eighteen previously acquired TGIF/TGIF2 photos supplied 90 matched deliveries,
including 36 known sharp or feathered edits. Select six sources per generator
family by original hash from the recorded held set after checking aspect ratio.
Resize each source and edit to the evaluated dimensions before delivery.
Include the original, local blur and local color adjustment as controls. For
each uploaded JPEG, `jpegtran -progressive` rearranged the coded coefficients
into ten scans. All 90 progressive decodes matched the baseline RGB bytes,
sampling factors and quantization tables exactly. Six separate paired model
runs also matched every model input and selected-region mask. These six cases
yielded no useful localized edit on either encoding, so they establish carrier
parity and supply no sensitivity estimate.
[Transcoding reference](https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/doc/usage.txt)
· [Source and licensing record](image-jpeg-signed-transitions-2026-11-12.md)

An additional progressive image with a published BG-REAL Stable Diffusion 1.5
insertion completed a simulated paid analysis. The first archived insertion
with any overlapping highlight was used for this integration check. Its
12,857-pixel mask covered 12,437 of the 15,245 known changed pixels. The browser
showed the same source crop, overlay and mask committed in the API, JSON and
signed PDF. The worker made one regional model call after simulated payment
confirmation. [External source terms](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)
· [Parity record](evaluation-panels/image-regional-progressive-2026-11-13.json)
· [Application check](evaluation-panels/image-regional-progressive-app-2026-11-13.json)

## Replay

The standard-library reviewer recomputes the totals and checks each decoded
image and model mask commitment in the ledger. The local run additionally
reopens all archived source pixels and uses the existing pinned model.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.image_regional_progressive_transfer review \
  --output docs/evaluation-panels/image-regional-progressive-2026-11-13.json
PYTHONPATH=backend OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  backend/.venv/bin/python -B -m scripts.image_regional_progressive_transfer run --output "$new_receipt"
```

The separate checkout test accepts the retained progressive JPEG with the
source digest in the application receipt:

```sh
cd backend
CHIPMUNK_REGIONAL_PROGRESSIVE_FLOW_INPUT="$retained_progressive_jpeg" \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/pytest -m slow \
  tests/test_native_regional_jpeg_paid.py -k progressive
```
