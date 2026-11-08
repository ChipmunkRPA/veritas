# Passage boundaries and mixed audio: source review

Reviewed 8 November 2026 while the independently frozen image-crop experiment
ran. This is a paper and selected-source inspection, not an executed detector
comparison or permission to deploy checkpoints. No third-party code, weights,
training data or remote model code was installed or executed.

## Passage localization: predict intervals directly

[GigaCheck's ACL 2026 publication](https://aclanthology.org/2026.findings-acl.213/)
uses a document backbone with a detection-transformer head to predict generated
text intervals. The work has an earlier 2024 preprint; the 2026 publication
does not make the underlying idea newly invented this month. The useful design
direction is learning boundaries with actual span labels, rather than expanding
a whole-document response into supposedly precise highlighted evidence.

The [repository at 8cdbe6a](https://github.com/ai-forever/gigacheck/tree/8cdbe6a1ce628b49b4aa0794edf7a425a47a774c)
has an Apache-2.0 code license. Selected files inspected:

| File | Git blob SHA | Observed mechanism |
|---|---|---|
| `gigacheck/inference/src/mistral_detector.py` | `3f1f5bcbbac0626f221eff43e20f1692cf81ce8d` | Length-limited token input; normalized intervals scaled to the processed text length; foreground scores and interval filtering. |
| `gigacheck/model/src/interval_detector/modules/matcher.py` | `3b7e340e8db6930cff2eda28edddb709d50a6305` | One-to-one assignment combining class, boundary-distance and generalized-overlap costs. |
| `gigacheck/model/src/interval_detector/losses.py` | `25905810ca6db82e11469c5eb07d8b5a743806ca` | Matched-span boundary/overlap loss and a separate no-object target for unmatched proposals. |
| `gigacheck/train/src/detection/interval_dataset.py` | `aec230882e198c6d1b4daa59eff6bec5f7a7aca2` | Cropped-text labels and character/token conversion; these coordinates need explicit round-trip qualification before use. |

The [released multilingual card](https://huggingface.co/iitolstykh/GigaCheck-Detector-Multi/blob/main/README.md)
declares Apache-2.0 and a Mistral-7B-v0.3 base. It documents English/Russian
training and warns about newer generators. This is not a completed audit of
all checkpoint, dependency and data terms. Its example enables remote model
code; this project did not execute that example. A 7B floating-point backbone
does not fit our unchanged 6-GiB local process limit. Quantization would be a
new implementation requiring its own parity and quality comparison.

For a compact independent implementation, preserve original Unicode character
offsets, explicitly mark unprocessed tails, learn interval/no-object outputs
from fitting-only span labels, and calibrate the new outputs independently.
Evaluate boundary overlap, missed generated characters, ordinary human passages
highlighted, and source-disjoint usefulness against the current passage check.
Model objectness is not calibrated probability of AI authorship. No training,
new point selection or serving replacement has begun from this review.

The separate [heterogeneous-span dataset card](https://huggingface.co/datasets/open-text-detector/heterogeneous-ai-spans)
offers mixed documents, matched controls and character offsets, but declares
mixed-source terms and records human-verification and requested/reported-model
identity fields. It requires source-level rights, lineage, prior-use and
annotation checks before any fresh-evaluation claim. No dataset files were
downloaded or scored in this review.

## Audio: distinguish speech origin from background origin

October's [What Do Deepfake Audio Detectors Hear?](https://arxiv.org/html/2610.17617v1)
studies independently genuine/synthetic speech and event mixtures. Its reported
component gap is substantially affected by an unseen speech domain; frequency
responses depend on the encoder. This supports testing each component's source
and generator separately. It does not establish that a frequency response
locates a synthetic speaker or that a synthetic event makes speech synthetic.

The [released source at 2d445b5](https://github.com/nii-yamagishilab/AntiDeepfake-AudioLLM/tree/2d445b525bb52518448b353fe1fee51d5f815af9)
separates its BSD-3-Clause code license from **CC-BY-NC-4.0 checkpoints**.
The latter are excluded from this commercial service. Selected source evidence:

| File | Git blob SHA | Observed mechanism |
|---|---|---|
| `LICENSE` | `854714abab16287e6959e761370bc0f8c56e8814` | BSD-3-Clause code terms. |
| `LICENSE-CHECKPOINT` | `b59f4c37c21279e91121ca5cf66ac517b7070e85` | Explicit noncommercial checkpoint terms. |
| `config/encoder_fc_classification.yaml` | `6f789adfb949624000d855a0d5e2e5acb6e8c264` | A fine-tuned encoder with a binary classifier and 16-kHz inputs. |
| `models/build_encoder_fc.py` | `4d1850805c6dbb8c6438ae69dd710b1c6e19628a` | Mean pooling followed by a linear classification head. |
| `models/output_heads.py` | `ab51f3481736d2058f27881e8fdcd278b1a24db5` | Distinct classification and verbalizer-score readouts; neither is our own held-out calibration. |

The README combines all three mixtures containing any synthetic component into
one fake class. That target cannot supply separate speech/background
probabilities for our reports. A next independent method needs separate known
component labels, retained original mixtures, unseen speech and event generators,
and ordinary noise/music/overlapping-speech controls. Component estimates require
their own calibration and a tested combiner. Existing spectral sensitivity and
separation results remain development evidence, not proof of sound authorship.

## Admission status

Both readings change the next methodological question, not current confidence.
The image experiment's frozen files and rules were untouched. Existing voice
calibration, rollback, prepaid USD9.99 flow, customer model-name filtering and
GCP deferral remain unchanged. Outside human review remains outstanding.
