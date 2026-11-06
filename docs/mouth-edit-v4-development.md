# AI mouth-edit evidence: development protocol v4

This study separates AI mouth editing from audio/video timing. The old geometric
timing trial is already consumed and rejected. A synchronized AI mouth redraw
is a positive AI edit here even when it has no audio offset. A constant 320 ms
audio delay is a non-AI control. Neither timing agreement nor a dark-mouth
correlation establishes authenticity.

## Rights and input construction

The official MediaPipe detector, Face Mesh V2 and blendshape model cards each
declare Apache 2.0. The existing `face_landmarker/float16/1` bundle was fetched
again from its official versioned URL and matched SHA-256
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
The already isolated MediaPipe 0.10.32 runtime is reused without replacing the
application's OpenCV package. It is a geometry input, never an AI detector.

Known AI edits use the publisher's human LivePortrait weights, HF revision
`82a4fa6735ca58432b6ce39301b4b9ee066dea47`, with MIT checkpoint metadata and
source revision `9b294b3d0536135442ea73cb01e6cb3ca7029dd3`. The source licence
explicitly requires replacing its noncommercial InsightFace detector for
commercial use. That component is excluded entirely; permitted mesh geometry
places the crop. All five human checkpoints and eleven imported inference files
were checked against publisher LFS/tree hashes. No animal, InsightFace, landmark
ONNX or X-Pose checkpoints are downloaded. These are publisher permission
records, not a training-data rights audit. The model is used only to create
local evaluation inputs, never a customer-facing detector.

Source footage is CREMA-D under its recorded ODbL 1.0 database/DbCL 1.0 contents
licences. Preserve publisher attribution and those separate terms for derived
metadata. Source media and model weights are not published in Git. This is a
development panel based on previously consumed actors, not fresh confirmation.

Select eight existing original clips with at least 63 complete common audio/video frames from the prior
28-actor timing study, by fixed SHA-256 order. Keep the first at most 100 frames
at the already matched 25 Hz and exactly corresponding mono 16 kHz samples.
Generation revision 4.1 bounds that span by floor(audio samples / 640) as well
as available picture frames. The first preparation found four recordings with
a partial final audio frame and retained 24 available/24 unavailable cases.
Its plan, receipts, source and protocol remain separate. This clock amendment
preserves all eight selected actor/source IDs and the original selection seed;
no detector responses have been inspected. Previously complete preparations may
be reused only with matching source, sample counts, generator assets, file and
mask hashes, explicitly linking their prior manifest. No speech is looped or padded.
Do not choose by detector responses. Every source produces six planned cases:
untouched picture/audio; local 12% brightening; non-neural crop/resizing;
constant 320 ms audio delay; AI redraw preserving the requested same-frame lip
ratio; AI retargeting using that source's ratio eight frames earlier. The last
two have known AI provenance; the requested ratio is not assumed to be the
achieved geometric measurement or an exact measured audiovisual delay.

Each visual edit occupies the middle 20 frames (0.8 s), with at least 21 frames
of surrounding context on each side. The retimed driver uses only earlier
source frames. Per-frame changes are blended inside a feathered mouth ellipse,
using two-frame edge opacity. Only that region is composited back into the
unchanged native frame. Both ordinary local edits use the same support and
temporal interval. Exact changed pixels are recorded for every edit frame.
The unmodified region must remain bitwise equal before encoding. All six cases
use the same lossless FFV1/PCM container recipe; decoded RGB, timestamps and PCM
must match the planned arrays exactly. Missing unique-face geometry or any
encoding mismatch remains an unavailable planned case, with no replacement by
response. Retain originals and before/during/after source positions.

## Intended comparison and freeze boundary

First verify local generation and source binding. Before scoring, freeze the
comparison implementation and recipe separately. The primary proposed change
uses the existing commercial YuNet and Apache-2.0 DINOv2 Small inputs, the same
native triplets/centres as face-sequence review, and a mouth-specific token
region. No extra-frame sampling or new authenticity checkpoint is bundled into
that comparison. The current face-sequence statistic is the primary localization
baseline. Record the current audio/mouth timing check separately; it cannot
serve as a calibrated AI-edit probability.

Choose a new mouth review point from development non-AI case maxima only,
just above the largest such response, without optimizing on AI examples.
Require at least three additional localized AI cases, at most one additional
non-AI review flag in each control condition, at most one lost previously
localized AI case, and no assessed-coverage reduction attributable to the new
statistic. Localization requires temporal IoU at least 0.2 with the known edit
interval; unassessed positives remain misses. All planned cases and both AI
variants are reported. A failed development result is rejected without consuming
fresh actors to rescue it. A passed result still needs a separately frozen
actor/source-disjoint confirmation, source-bound evidence integration, UI/PDF/JSON
and paid-flow checks before local enablement. It must retain zero weight in
overall confidence until separately calibrated for that purpose.
