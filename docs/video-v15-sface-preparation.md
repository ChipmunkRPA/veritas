# Bounded face-swap transitions: permitted feature model acquired

7 November 2026. Acquired a small SFace/MobileFaceNet feature extractor as a
possible ingredient for a different transition-localization method. The official
[OpenCV Zoo model-directory notice](https://github.com/opencv/opencv_zoo/blob/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_recognition_sface/README.md#license)
explicitly applies Apache2.0 to **all files in that directory**, including its
stored ONNX model. The exact per-model license and attribution are retained.
This is a weight-inclusive declaration, not an assumption based on a separate
code repository's license. Full training-data rights are not independently
established by that declaration.

Repository revision: `48b10fa505aeb51fbd127fc60a5c660040e823b7`.
Weight bytes:38,696,353. SHA-256:
`0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`.
The complete ONNX file matches its publisher Git LFS commitment. README,
per-model LICENSE and reference wrapper match their Git blobs. The reference
Python was inspected but never executed; no inference runtime was initialized,
new package installed or project media scored. Weights remain outside Git.
[Acquisition receipt](evaluation-panels/video-v15-sface-acquisition.json).

## Hypothesis, not a new detector result

Current bounded-edit checks confuse some ordinary appearance/pose changes with
manipulation. A pose-tolerant face feature may help compare a stable track
before, during and after a short edit: similar surrounding appearances with a
different middle appearance could support review of a bounded transition.
This requires local measurement, ordinary-change controls and known boundaries;
the face-recognition benchmark numbers do not validate deepfake detection.

SFace is **not an AI classifier or lip-sync detector**. It cannot establish that
a visible change was AI-generated, and a consistent whole-clip face swap may
have no temporal transition to detect. No named-person identification or
reference identity database is part of this proposal. Model distances must
remain feature measurements, not invented probabilities or customer confidence.
Do not adopt the publisher's identity-match thresholds for forensic decisions.

First inspect the declared graph/runtime contract, verify the standard local
OpenCV API with synthetic inputs, and check exact five-landmark alignment,
RGB/BGR preprocessing and source/crop/clock correspondence. Heavy model
comparisons remain sequential with the active audio work. Freeze a complete
consumed-source comparison before scoring project clips, retaining all prior
false-flag, usefulness and localization requirements. Passing development is
required before any separately frozen new-source confirmation and UI/PDF/JSON
integration or overall-confidence consideration.

Keep actual face features private; no biometric feature packet is published by
this preparation. Source/model terms, mislocalizations, misses and ordinary-edit
false alarms must remain explicit. The existing model, rollback, customer
evidence, billing and GCP infrastructure remain unchanged. No accuracy gain,
calibration or outside human review is claimed.
