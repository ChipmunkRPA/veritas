# Alternative writing detector: acquired, not evaluated

7 November 2026. Downloaded a standard ModernBERT-base classification checkpoint
for a future local comparison. No model construction, classifier forward on
project text, fitting or serving change has taken place. The resource-heavy
audio extraction continues alone.

The [additive comparison implementation](writing-modernbert-v4-execution.md)
and [prospective protocol](writing-modernbert-v4-protocol.md) are now available.
Synthetic runtime verification and the completed execution freeze are pending;
the code release does not claim that either has run.

## Source and permitted use

The [exact fine-tuned card](https://huggingface.co/GeorgeDrayson/modernbert-ai-detection-raid-mage/blob/e047d9c56f51a7e5c5ef5b0dd556f519bb9dd624/README.md)
declares Apache-2.0. Its [exact base card](https://huggingface.co/answerdotai/ModernBERT-base/blob/8949b909ec900327062f0ebf497f51aef5e6f0c8/README.md)
explicitly releases model weights, architecture and training code under
Apache-2.0. Retain attribution and applicable notices. This supports commercial
evaluation/use under those terms, not a warranty of all training-data rights or
future detection accuracy.

Fine-tuned revision: `e047d9c56f51a7e5c5ef5b0dd556f519bb9dd624`.
Weight file:598,439,784 bytes; SHA-256
`2ae8059c621795fc490353bbd920bc3672e6eaf8ad66ddb7c015b4ecb0129218`.
All five configuration/tokenizer/card assets and both base metadata files match
their publisher Git blob identities. The complete safetensors file matches its
publisher LFS digest. No base checkpoint, pickle, custom remote code or training
package was acquired or run.

Static inspection finds138 contiguous F32 tensor entries,149,606,402 stored
values, a2×768 classification weight and two scalar biases. Configuration names
the installed standard `ModernBertForSequenceClassification`, mean pooling and
a native8192-token positional limit; it declares no custom-code mapping. These
are file/layout observations, not a successful strict architecture load or
validated8192-token detector performance. Weight files remain outside Git.
[Acquisition/static receipt](evaluation-panels/writing-v4-modernbert-acquisition.json).

## Interpretation and future comparison

The publisher reports MAGE and RAID training. Those sources cannot be presumed
unseen validation, and exact full pretraining overlap remains unknown.
The [pinned inference source](https://github.com/GeorgeDrayson/model_collapse/blob/feb8511479a2e2dc868e1caf3f63cb99f1fcc746/src/utils/detector.py)
uses two-logit softmax and positive index1. The saved config also contains
`problem_type=multi_label_classification`. Retain that metadata; inference
without labels follows the publisher's softmax convention rather than changing
it into an independently calibrated authorship probability.

The publisher configuration includes temperature1.30173 and point0.74938 for
its own task. Neither is adopted. Its left truncation, optional compiler-error
suppression and threshold do not substitute for complete input coverage,
explicit failures and our separately frozen comparison.

First strictly construct/load the standard local model after the audio run,
then check deterministic synthetic single/batch responses, positive-class
mapping and complete tokenizer overflow identity. No remote publisher code
will be executed. Publish the complete comparison implementation,
source/model/runtime commitments and protocol before scoring any project text.

Use the consumed40 mixed documents/674 exact foci for development, retain the
context/focus2×2 design, both existing raw-response reference points and all nine
original quality requirements. A different architecture is a hypothesis, not
evidence of improvement. Current rewritten/humanized-text support remains part
of the comparison's interpretation; a binary generation head cannot inherit
those claims automatically. No raw response becomes word-origin proof.

Only passing development can justify a separately frozen unused-author test
using the [audited human controls](writing-v4-human-controls.md) and unused mixed
documents. Preserve all unresolved source cases and uncertainty. The rejected
TMR candidate is not retested on these controls to rescue its result. Own
calibration, modern-generator coverage, source-position evidence and complete
customer/PDF/JSON/prepaid checks remain necessary before adoption or overall
confidence change. This preparation establishes no accuracy result or outside
human review. GCP remains last.
