# Background context for face transitions: prospective development protocol

Ordinary source substitutions produced 22 appearance alerts in 40 clips in the
[v24 diagnostic](video-genuine-edits-v24-result.md). Those alerts locate real
appearance changes but cannot establish AI authorship. This new comparison asks
whether contemporaneous background measurements help separate face-local change
from whole-scene transitions. It does not assume background stability proves a
deepfake, infer pixel authorship, or fit a customer AI confidence score.

## Fixed method and controls

Keep the same source-bound SFace appearance responses and temporal supports from
v20. SFace weights remain the pinned Apache-2.0 asset with SHA-256
`0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`.
No new checkpoint is downloaded or selected. The first eligible hash-ordered
clip in each of the two source roles repeats every recorded SFace alignment,
feature and supported appearance window exactly before other cached values
are reused. The remaining old neural responses are reused, not re-inferred.

Divide each actual 640x480 RGB frame into 12x16 cells of 40x40 source pixels.
Compute float64 cell/channel means divided by255. On each fixed temporal window,
exclude every cell intersecting any source face box expanded25% on each edge,
using one common mask over all its frames. Require at least25% background cells;
otherwise the contextual response is unavailable. Do not fill it with zero.

Keep the original three middle centers, two before and two after centers, and
three consecutive source frames per center. Average the cell descriptors in
these groups. On common background cells, let L and R be mean absolute RGB
differences between middle and the two flanks; F is the flank difference.
Background change is max(0,min(L,R)-.35F). The fixed primary response is the
original appearance response minus twice this background change. The multiplier
matches cosine-distance range2 to normalized RGB range1; it is a predeclared
engineering hypothesis, not a learned causal attribution or proven optimum.

Compare three arms: the unchanged raw appearance point, appearance with its own
matched genuine-only point, and the contextual primary with its own point.
The second arm separates recalibration from the new information. No coefficient,
mask size, resolution, checkpoint or successful subgroup is selected afterward.

## Exact sources, point selection and diagnostic denominators

Use80 genuine fusion_cal cases in40 existing connected source groups to choose
points. The80 AI fusion_cal cases are not planned or scored. Within each common
assessed genuine group take the maximum over original/ordinary cases. Require
at least25 such groups. Count flags at the unchanged raw appearance point
.11243900750879454, then choose each new point at the descending value whose
strict exceedance count is no greater than that same budget. Exact ties do not
flag. Wholly unassessed groups remain in the availability denominator.

Freeze both fitted points and the complete genuine journal before any new
diagnostic measurement. No diagnostic or AI response selects either point.
The fit-completion receipt binds point, journal and prospective-freeze hashes;
later phases refuse a changed point file before reading diagnostic pixels.
These are internal comparison points, not probabilities or public thresholds.

Diagnostics contain all160 previously consumed confirmation cases:80 swaps,
40 originals and40 ordinary controls, plus all40 v24 copy-only scene substitutions.
The latter link the40 confirmation source groups into20 paired components;
carry those links through all diagnostic component counts. Calibration sources
remain disjoint. This is consumed development data and not fresh people,
generators, independent truth or an unseen holdout. Retain every planned file,
unavailable unit, error and miss. A partial/empty observation cannot clear a file.

Merge flagged temporal supports with the existing550ms gap rule. Compare best
interval IoU to exact known edit bounds; IoU>=.3 is useful localization. Report
mean IoU including misses, edited-file alerts, original/ordinary/scene-cut flags,
connected-component counts, each generator/control family and unavailable cases.
All200 diagnostics appear in each of the three arms, totaling600 outcomes.
Scene-cut ranges are ordinary edits, not correctly identified AI intervals.

The primary must improve useful-AI counts and mean IoU over both references,
lose no previously useful edit, add no false-flag components in each genuine
condition, reduce the scene-cut flag count and add no wholly unassessed file.
A failed reference or family cannot be omitted. Passing development would still
require fresh source-separated confirmation and full-app, timestamp/frame,
PDF/JSON and prepaid-flow verification before any local adoption. The cue does
not enter overall confidence in this experiment.

## Native binding, independent arithmetic and publication

Publish/verify this protocol, code and input commitments before new project
pixel measurements. Reconstruct exact prepared media bytes and native PTS;
match every used frame to both the earlier decoded-source RGB commitment and
cached face-model input. Every new block mean is checked independently through
integer integral sums within1e-12, and every window through a scalar cell-wise
transcription within1e-12. Store new arrays and face geometry privately. A
separate process reconstructs arrays, locations, points, intervals and all
outcomes. It does not independently repeat every old face-model forward.

The old v20 repository-wide freeze remains immutable. Current unrelated app
files have evolved; this study does not weaken or claim a pass of its old
whole-repository check. It binds the unchanged source/decoder/appearance helpers
it actually executes and the original data, journals, weight and runtime pins
in its own prospective freeze. Source construction checks stay unchanged.

Run source point, diagnostic, ordinary reconstruction and audit sequentially,
each with900 seconds,6GiB sampled process-tree RSS and at least1536MiB free
RAM/disk. Retain guard/monitor errors and failed temporaries; no implicit restart,
case replacement or response-driven amendment. Sampled memory is a lower bound.
Publish code, commitments and complete aggregates only. No raw photos, clips,
face/background vectors, pairings, customer uploads or private weights are
published. Outside human review, serving admission, cloud resources and changes
to billing or speech calibration are not implied by this study.
