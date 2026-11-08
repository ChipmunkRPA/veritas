# New localizers and word-edit methods: source review and executed contracts

This prepares the next detector comparison. It does not establish improved
accuracy, calibrated confidence or a serving release. Neither new image model
has scored project media. All checks ran locally; GitHub Actions remains disabled.
Current confidence, rollback, prepaid USD9.99 analysis and GCP deferral are unchanged.

## Word editing needs paired provenance and separate targets

September's [MixDetect paper](https://arxiv.org/html/2609.32625v1) separates edit
occurrence from conditional intensity. Training uses source/edited pairs,
lexical anchors and ordered contextual alignment; inference needs only the
resulting text. Its contextual encoder is fine-tuned. Our v7 frozen-encoder
interval head is not a reproduction of that system. A semantic difference alone
cannot identify an editor as AI.

This matters after [v7](writing-spans-v7-result.md): stronger multi-region coverage
did not preserve larger inserts, while every short insert remained missed.
Training a word-edit system needs genuine paired provenance and ordinary
human-to-human edits. Unrelated answers to one question cannot supply reliable
edit-intensity labels.

The paper's principal paired corpus comes from
[EditLens](https://huggingface.co/datasets/pangram/editlens_iclr), whose publisher
identifies the collection as noncommercial. Its
[released classifier](https://huggingface.co/pangram/editlens_roberta-large)
also declares CC BY-NC-SA4.0. Neither data nor checkpoint was acquired for this
commercial project; mirrors and a paper description do not create permission.

September's [unified image-forensics paper](https://arxiv.org/abs/2609.02640)
separates genuine/generated/tampered classification from segmentation. The
[repository](https://github.com/anngal01/From-Detection-to-Localization-A-Unified-Forensics-Framework-for-Fully-Synthetic-and-Tampered-Images)
has MIT code and retained Apache-2.0 portions. Its pretrained-weight link is
commented out in the README. Neither that link nor a code license qualifies
the weight files; none of its code or checkpoints was executed here.

## Small whole-image classifier: actual execution, limited spatial coverage

The [imdet publisher](https://huggingface.co/Deckard618/imdet) explicitly applies
Apache-2.0 to the PE-tiny folder's model and config. The upstream
[Perception Encoder card](https://huggingface.co/facebook/PE-Core-T16-384) agrees.
We acquired only its12,509,961-byte checkpoint at
`17e493fd263f9195999d01dde1eee78ea8956638`, retaining full terms and notices.
The [acquisition](evaluation-panels/image-imdet-v17-acquisition.json) pins the
publisher's exact LFS hash; its advertised accuracy is not our validation.

The [graph inspection](evaluation-panels/image-imdet-v17-graph.json) validates
2,274 standard ONNX nodes and168 finite initializers:6,013,954 elements in
float32/float16. No custom domains, external tensors or nested execution graphs.
The [native receipt](evaluation-panels/image-imdet-v17-authored-native.json)
records18 actual CPU forwards on six authored RGB patterns. Independent
Pillow/NumPy inputs exactly match the published Torchvision preparation;
canonical repeats are exact. Optimized-path maximum differences are5.36442e-6
logits and8.39278e-7 raw response, inside the predefined.01/.001 limits.

Nineteen tests cover source bounds, changes to actual tensors, conservative
bilinear support and invalid-input refusals. A wide authored source has only
12.586% nominal center-crop coverage. These coordinates describe inspection,
not an edit mask or pixel-authorship probability. The classifier cannot supply
an originality verdict for unseen image areas. Training overlap remains unknown.
Runtime:3.747s,406,093,824 bytes sampled tree RSS, no sampling errors or guard.

The isolated runtime installation initially stopped at its conservative disk
reserve check before installing ONNX. That failed attempt/log remains. A
separate attempt installed only the missing package after free disk increased;
no model/research artifact was deleted or app dependency modified. One earlier
missing-module invocation started no acquisition; the corrected invocation did.

## RelayFormer: full checkpoint and geometry contract

[RelayFormer](https://arxiv.org/abs/2508.09459) has a2025 preprint and a current
release; it is not newly invented in November. Its
[source at23c973f](https://github.com/WenOOI/RelayFormer/tree/23c973fc956195537d85148d124d6a146243f6c9)
is MIT, while the official [checkpoint card](https://huggingface.co/Wenn11/RelayFormer)
separately declares Apache-2.0. The acquired image-only checkpoint is1,102,625,388
bytes at `348c5f7c9069336846cd74bdbbd5cdc85693555d`; source, notices and full
checkpoint hashes are in the [acquisition](evaluation-panels/relayformer-v18-acquisition-v2.json).
Independent clearance of training photographs is not established. Video and
application-oriented weights were not acquired or substituted.

An earlier acquisition stopped before weights because our literal card check
expected an extra newline. Its program, downloaded source and failure receipt
remain private. A separately recorded successor checks the exact publisher-card
SHA256 and uses new paths; it overwrites no original attempt.

[Weights-only inspection](evaluation-panels/relayformer-v18-checkpoint.json)
checks410 finite float32 tensors,91,909,179 model elements. Only the standard
`argparse.Namespace` metadata container is additionally allowed. Optimizer,
epoch and scaler data are unused; downloaded model code does not execute in
that inspection.

Before any authored forward, the reviewed graph loads with base downloads
disabled and networking denied during construction. An identity registry shim
replaces only unused training registration. The non-strict/unsafe base-loader
path is skipped: every full-checkpoint state field, shape, dtype and tensor
value must match, with strict loading. Initialized values cannot fill missing
weights. The separate MIT rotary helper is pinned0.8.9 with its official wheel hash.

The sample image loader crops oversized inputs. Our adapter instead preserves
the full image, resizing only above1024, then explicitly pads/normalizes. The
sample video loop can drop trailing incomplete clips; a future video adapter
must retain real timestamps/tails and distinguish padding. No new video
inference or video-coverage fix is claimed.

The [authored full-model receipt](evaluation-panels/relayformer-v18-authored-native.json)
records six actual CPU forwards on two patterns, including an oversized source.
All410 tensors load exactly. Actual native patches match source rectangles;
repeat maps are exact. Merged/unmerged maximum map difference is1.63913e-6,
inside the predefined3e-4 bound. Thirteen tests cover full source coverage,
integer-ratio projection, padding exclusion, orientation and missing tensors.
Runtime:11.361s,1,768,062,976 bytes sampled tree RSS, no sampling errors or guard.

Source maps retain float32 values and use exact integer center indices; no
uint8 map quantization is substituted. Image scaling and neural receptive
fields still prevent pixel-authorship claims. The scaling adaptation is
explicit and is not a claim of literal paper replication or real-edit accuracy.

## Qualification still required

Freeze real-image source roles, retained maps, genuine/ordinary controls,
baseline bindings and rules before project-media inference. Compare useful
overlap, misses, false-highlight area and source-group flags in native/JPEG
formats. Complete source/native replay and a separately frozen fresh
confirmation are required before release. Localization alone cannot supply
overall AI-origin confidence. Original failed audits remain failed.

Public authored geometry checks require no weights, photos or credentials:

```sh
PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 backend/.venv/bin/pytest -q -o addopts='' backend/tests/test_image_imdet_v17.py backend/tests/test_relayformer_v18.py
```

For a permitted native review, use the public acquisition helpers in a fresh
checkout, fresh receipt filenames and separate pinned research dependencies;
committed metadata must not be overwritten. Native modules run through the
local resource guards with fresh output/profile paths. Return your revision,
environment, exact receipt and scope via the existing independent-evaluation
issue form. Do not run GitHub workflows. Outside human reviewers remain0.
