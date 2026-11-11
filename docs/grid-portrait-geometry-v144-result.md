# Fresh pilot geometry completes without exclusions

All **16 fixed pilot clips** completed their separate guarded CPU stages.
The 448 distinct native-picture measurements and 16 fixed repeat inputs produced
usable geometry. Every own repeat matched exactly, including mesh coordinates,
crop mappings, source aperture, crop PNG and float mouth-support array. No clip
or frame was replaced or excluded. The sixteen reserve clips received no face
inference in this stage.

All workers exited zero with no guard violations or sampling errors. Summed
worker-stage duration was **64.94 seconds**, with stages ranging from 3.18 to
13.46 seconds. Maximum sampled descendant RSS across stages was
**505,233,408 bytes (0.47 GiB)**. Samples do not guarantee the true memory peak.
The unchanged neural limits and initial/running reserves remained in force.
See the [aggregate execution receipt](evaluation-panels/grid-portrait-geometry-complete-v144.json)
and [frozen method](grid-portrait-geometry-v144-method.md).

The count of 464 is **face-task API inputs**, not independently counted internal
neural graph invocations. No generator, speech encoder, identity recognizer or
deepfake detector ran in these stages. The new code passed 43 authored tests,
Ruff and diff checks before publication and execution. Saved completion replay
also reconciled every actual request/status journal and resource profile.

Exact source, runtime and model-card commitments were checked before inference.
The original primary model cards declare Apache-2.0 for
[BlazeFace](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf),
[FaceMesh V2](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf)
and [blendshapes](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf).
Retained runtime files and source/card bytes are bound separately from dataset
terms; the task supplies geometry, not an authenticity judgment.

This completes original-picture geometry preparation for the 176-case pilot.
Generated edits, ordinary controls, achieved visible motion and the frozen-head
comparison still require actual execution. Successful geometry is not a new
detection-accuracy or customer-confidence result. All raw media, face vectors,
individual fingerprints, pairings and clocks stay private; publication contains
code, aggregates and whole-artifact commitments only.
