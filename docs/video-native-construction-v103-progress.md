# Full known-boundary video construction is running locally

The remaining 89 groups of the existing
[native-insert plan](video-native-inserts-v95-method.md) are now executing
sequentially on the approved Seagate development archive. The earlier
[engineering pilot](video-native-insert-pilot-v96-result.md) is retained without
rerunning it. No source group, role, construction formula, codec setting or
resource limit has changed.

At the [published start snapshot](evaluation-panels/video-native-construction-start-v103.json),
**three groups, 18 clips, 2,880 native RGB frames and 54 before/during/after PNGs**
passed both construction verification and the second command-interface review.
This includes the original pilot and two newly completed groups. The exact
owned controller process was inspected and was live when the snapshot was
recorded. This is a point-in-time progress receipt, not a terminal completion
claim or evidence that a later process is still running.

## What the complete panel will contain

The frozen plan contains 90 compatible groups, 540 clips, 102,312 native output
frames and 1,620 representative PNGs. Sixty-one groups retain 1920 x 1080 source
pictures and 29 retain 1280 x 720. Observed native clocks and frame counts vary;
each clip uses its own actual rational presentation times. Six incompatible
source groups remain explicit in the parent inventory.

Each compatible group supplies an unchanged original, two face-swap insert
conditions, and ordinary lighting, blur and timing controls. The middle-third
edit and the original context on either side are bound to their exact source
frames. The captures show known construction positions, not regions located by
a detector. RGB differences are not exact AI-pixel truth or independent
verification of a changed face identity.

## Execution and verification

The unchanged v95 worker encodes and decodes every intended output frame,
requiring exact RGB bytes and rational timestamps. After each successful group,
the unchanged v96 reviewer reads every frame through FFmpeg framehash, checks
stream dimensions and display duration through ffprobe, and verifies all
representative PNG bindings and source correspondence. Only then does the
queue advance. Both paths share the underlying codec library; this is a
separate command implementation, not an independent codec engine or outside
human review. The review does not independently reimplement the lighting and
blur transformations.

All 23 relevant constructor/reviewer tests passed locally before launch, as did
an owned-child process-monitor permission preflight outside the sandbox.
Completed groups have zero resource guards and sampling errors. Their largest
sampled construction descendant RSS is 828,817,408 bytes; the largest observed
review parent/command RSS is 638,697,472 bytes. Samples are not guaranteed peaks.
Each stage retains its original 900-second/1.5-GiB data-only bounds, initial
3-GiB RAM/disk requirements and 1.5-GiB running reserves. No neural models or
generators are executed; existing neural-study limits remain unchanged.

The controller starts only previously unattempted groups, in the frozen order,
and preserves the successful pilot. It records each exact child exit and stops
on a failed construction or review. No automatic retry, overwrite or implicit
resume is allowed. MP4s, PNGs, source identities and per-frame fingerprints
remain private. Only aggregate progress and whole-artifact commitments are
published. The source permissions and missing portrait-attribution limits
remain those in the [original preparation](video-common-source-preparation-v89.md).

## Detection work remains separate

These silent inputs support the upcoming shared comparison of current crop,
appearance and consecutive-frame checks, localization overlap and ordinary-edit
false alarms. They do not establish new detector accuracy or change customer
confidence. New fitting/calibration and held source roles must remain separate.

Lip-sync work uses the separate recorded audiovisual panel. Its known sentence
overlap across roles, historical sentence contexts, short usable support and
audio/video origins still need an explicit construction/split solution before
training or testing a new articulation method. Silent swaps cannot validate it.

No serving default, billing behavior or GCP deployment changes here. Publication
is by reviewed manual push after local verification. No GitHub Actions or
external outreach is used.
