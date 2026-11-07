# Image classification candidate: assets verified, inference pending

7 November 2026. Acquired Smaug as a separate **whole-image** candidate, not a
replacement for edit-boundary localization. The [pinned detector notice](https://huggingface.co/abacusai/Smaug-AI-Image-Detector/blob/c7c95951e04a87b884a4ad479401b1128c3903bd/README.md#8-license)
and [exact Apache2 license](https://huggingface.co/abacusai/Smaug-AI-Image-Detector/blob/c7c95951e04a87b884a4ad479401b1128c3903bd/LICENSE)
cover the released detector; the [pinned EmbeddingGemma2 base notice](https://huggingface.co/google/embeddinggemma-2/blob/f9972ac22e696a105c482202646979e33e4b5a1e/README.md)
also declares Apache2. This establishes the publisher grants, not independently
cleared training-image rights or performance on our sources.

The877,599,816-byte base weight and32,170,510-byte tokenizer match publisher LFS
commitments. Header inspection verifies624 contiguous complete BF16 tensors,
438,760,472 values, plus the separate769-value classification head. No neural
runtime was initialized or project image scored. Model files stay outside Git.
[Acquisition snapshot](evaluation-panels/image-v7-smaug-acquisition.json) and
[base notice commitments](evaluation-panels/image-v7-smaug-base-notices.json).

The app's Transformers5.15.1 lacks this architecture. Four compatible,
hash-verified wheels were installed into a separate research environment:
Transformers5.22.2, SentenceTransformers6.2.0, Hub1.32.0 and Tokenizers0.23.1.
Other existing dependencies are referenced read-only. Their declared dependency
requirements pass metadata checks, but actual model loading remains untested.
App packages stay unchanged; all79 frozen audio source files and six critical
installed audio implementation hashes were rechecked unchanged.
[Research environment receipt](evaluation-panels/image-v7-smaug-research-environment.json).

## Actual source preprocessing check

The research crop implementation records the EXIF-oriented source size, exact
resize/crop bounds, rational source-edge coordinates and resampling method.
All256 crop RGB outputs match the publisher on128 self-created cases spanning
eight shapes, all eight EXIF orientations, and single/three-view modes.
Two local executions agree. Only the exact hash-pinned, inspected pure-PIL crop
function was executed; the publisher's full module, imports, device selection
and model loader were not. [Complete synthetic receipt](evaluation-panels/image-v7-smaug-native-views.json).

This proves crop/orientation correspondence, not processor tokens, neural
outputs, authenticity or a suspicious pixel mask. A high response for a crop
does not prove its exact edited pixels. Overlapping/identical crops are not
independent observations. The native model/pooling/normalization/head and
mean-logit reduction still need actual source checks before a complete frozen
baseline comparison. Independent calibration, useful detections, genuine-image
false alarms, compressed/ordinary-edited controls and separately frozen unused
sources remain required. No publisher accuracy or operating point is adopted.
The existing service, overall confidence, billing and GCP remain unchanged.

Acquire the exact pinned reference source separately, then reproduce only the
synthetic view check without a model download:

```sh
PYTHONPATH=backend python -m scripts.audit_smaug_views_v7 \
  /path/to/pinned-smaug_image_detect.py /path/to/new-receipt.json
```
