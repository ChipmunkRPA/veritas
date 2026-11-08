# Background context: small control gain, replacement rejected

The frozen primary **failed the replacement rules and remains disabled**.
It reduced individual scene-substitution alerts from 22 to 20 of 40, but all
three arms still flagged 11 of 20 connected scene-substitution components.
The one extra localized swap versus the fixed reference was also recovered by
recalibrating appearance alone. Background information found no additional swap
beyond that control, and the primary added a jump-cut alert versus the fixed
reference. No serving or overall confidence change follows.

The [protocol](video-background-context-v25-protocol.md),23 source-code pins and
[input freeze](evaluation-panels/video-background-context-v25-freeze.json) were
published as [cf62bee](https://github.com/ChipmunkRPA/veritas/commit/cf62bee345c524862bdd962db87c7fafe1f5de9f)
and verified remotely before new project pixels were measured. All 280 planned
files completed:80 genuine point files and 200 diagnostic files. These are
consumed source groups, not fresh people, generators or independent truth.

## All comparison outcomes

Each arm retains all 80 swap clips,40 original controls,40 existing ordinary
controls and 40 newly measured copy-only scene substitutions. Useful localization
means best merged interval IoU>=.3 with the known edit; misses count as zero.
Appearance alerts are inspection evidence, not calibrated AI authorship.

| Method | Swaps flagged /80 | Useful localization /80 | Mean interval IoU | Original alerts /40 | Existing ordinary alerts /40 | Scene-substitution alerts /40 |
|---|---|---|---|---|---|---|
| Current fixed appearance reference |62|62|.562931|0|1|22|
| Appearance with matched point |63|63|.575344|0|2|22|
| Background-context primary |63|63|.576732|0|2|20|

The primary misses17/80 swaps, versus 18/80 fixed and 17/80 recalibrated. These
include unavailable observations and are not original-file verdicts. No
previously useful swap was lost. Against the fixed reference, the primary fails
the no-extra-jump-cut-component rule. Against recalibrated appearance it fails
the more-useful-swaps rule. Higher IoU alone does not satisfy that requirement.
There is no post-result coefficient change, point selection or subgroup adoption.

| Family | Planned files | Fixed alerts | Recalibrated appearance alerts | Context alerts | Wholly unassessed in each arm |
|---|---|---|---|---|---|
| Inswapper swaps |40|32|32|32|1|
| Uniface swaps |40|30|31|31|2|
| Original videos |40|0|0|0|1|
| Lighting controls |15|0|0|0|0|
| Recompression controls |12|1|1|1|1|
| Jump-cut controls |13|0|1|1|0|
| Copy-only scene substitutions |40|22|22|20|12|

The scene-substitution component count remains11/20 in all arms. The new
jump-cut alert adds one flagged component in that condition. A scene change
can legitimately trigger an appearance alert; interpreting that as AI origin
would be unsupported. The source construction introduces no new AI but relies
on publisher-original labels for its parents, not independently verified history.

## Coverage, sources and point isolation

All three methods have17/200 wholly unassessed files and 6/200 partially
assessed files;177 have all planned windows assessed. They assess891 of 1,000
planned diagnostic windows. Those counts describe the frozen sampled supports,
not full-frame continuous coverage or complete video authenticity. A partial
file can show an observed positive change while other windows remain unknown;
absence of an alert cannot clear its unexamined content. Every exported research
outcome retains a null originality verdict. The earlier v24 complete-required
summary withheld a file when any window was unavailable. Its 18 scene-control
inconclusive cases split here into 12 wholly unassessed and six partial cases;
that breakdown does not upgrade any partial case to an originality finding.

The source substitutions join the 40 consumed diagnostic source groups into 20
paired components, which are used in component counts. All 80 genuine point
files remain in their disjoint40-group role. The two new points use 39 commonly
assessed groups; the other group stays in the availability denominator. Both
points match the fixed reference's one flagged group in that common set.

| Method | Internal ranking point |
|---|---|
| Fixed appearance |.11243900750879454|
| Matched appearance |.10179733083341733|
| Background context |.09302866964157606|

The point-fit completion receipt was saved before diagnostic pixels were read;
later phases checked its point, journal and prospective-freeze hashes. No AI
calibration file or diagnostic response selected a point. These values are not
AI probabilities or public screening thresholds. The shadow artifact's
`unassessed_groups: 0` refers only to its prefiltered39-group common set. The
public result explicitly retains the original40-group coverage:39 common and
one unavailable. It does not silently turn that excluded group into an assessment.

## What the new measurement tests

The new descriptor uses exact40x40 source-pixel cell means on a12x16 grid,
excluding all expanded face boxes on one common temporal mask. It compares
the same three middle and two-before/two-after groups as the existing cue,
including their consecutive source frames. The primary subtracts twice the
background transition from the original appearance response. Grid size, face
margin, availability rule, temporal arithmetic and multiplier were fixed before
source scoring; no new classifier or checkpoint was selected.

The existing [SFace model documentation](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/README.md)
licenses all files in its directory under Apache-2.0. The exact retained weight
hash was checked. In the first eligible clip of each source role, all 33 recorded
face inputs were re-executed:66 new SFace forwards, with exact alignment,
feature and window-response parity. Other source-bound face responses were
reused. This is not a new inference run of every old face model.

All 9,240 newly measured source frames match both their earlier decoded RGB
commitment and cached face-input commitment, with exact native clocks. Block
means were independently checked inline via integer integral sums; the largest
difference was1.110223e-16, below 1e-12. Cell-wise scalar temporal arithmetic
differs by at most5.551115e-17. The separate review replays every saved private
array/window, common mask, both genuine-only point refits, source interval and
all 600 three-arm outcomes. It does not perform a second full pixel extraction
or repeat every old neural forward.

## Resources and actual separate-host checks

| Phase | Seconds | Sampled process-tree RSS lower bound, bytes | Sampling errors | Exit / guard |
|---|---|---|---|---|
| Genuine points |122.033|1,088,651,264|7|0 / none|
| Diagnostics |146.602|1,046,036,480|12|0 / none|
| Ordinary substitutions |203.935|1,418,117,120|9|0 / none|
| Complete saved-array audit |5.244|209,272,832|0|0 / none|

Each ran sequentially within its 900-second/6-GiB guard, with minimum available
RAM/disk checks. Process sampling errors are retained; RSS is a sampled lower
bound, not an exact peak. No source-scoring phase failed or was implicitly
restarted. Private before/during/after illustrations use six already committed
source anchors and remain local; they show both a corrected control and a
remaining alert, not just successful examples.

All 18 focused local tests pass. The actual
[Linux run](https://github.com/ChipmunkRPA/veritas/actions/runs/42481140674) passed
the same 18 authored-array tests, with its
[decoded-log-bound receipt](evaluation-panels/video-background-context-v25-linux-array-review.json).
It acquired no model, video or project descriptor. It tests exact block
arithmetic, face-only versus scene interventions, masks, missing context,
genuine-group points, intervals and point-file tamper refusal. It does not
establish native detection quality or constitute outside human participation.
Outside human reviewers remain zero.

The first local isolation harness failed before tests because replacing the
socket constructor disrupted SSL class import. A corrected external harness
passed without changing frozen science. The first GitHub dispatch returned404
before workflow registration and created no run; an authoritative active-workflow
check preceded successful dispatch. Those events remain in the
[complete public aggregate](evaluation-panels/video-background-context-v25-result.json).

## Consequence for the next method

Background colour context removes a few obvious source substitutions, but
appearance plus coarse surroundings still cannot reliably distinguish ordinary
edits from AI manipulation. The experiment does not justify admitting this
candidate or its recalibrated shadow. A stronger method needs manipulation-
specific spatial/temporal evidence and controls, with the same source-bound
timestamps and an independently evaluated combination. A new development freeze,
untouched source confirmation and full-app/PDF/JSON/prepaid-flow verification
remain before replacement. Raw clips, face/background descriptors, source
pairings and learned private weights were not published. Serving speech
calibration, model defaults, billing and GCP resources were unchanged.
