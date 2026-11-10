# Blind native mouth and speech features

This is a new feature experiment on the completed v120/v121 native-time bank:
283 ready clips and five explicitly unavailable edit conditions across all 48
source-connected groups. Preserve the fixed 18 fitting, ten calibration, ten
selection and ten confirmation roles. No classifier, calibration or detector
accuracy is produced in this stage. The v6 aligned-face crop and its old fitted
head are not reused: only its frozen pixel preprocessing and feature pooling are.

## Inputs and representation

Decode every native picture of every ready condition. Verify the complete input
file hash and each decoded RGB hash against the previously audited journal.
These integrity hashes do not select pictures or enter the numerical features.
Detect one eligible face with the cached MIT YuNet, minimum face side 48 pixels.
Use the centre of the two native mouth corners; crop horizontally plus/minus
0.9 and vertically plus/minus 0.6 times their Euclidean distance. Floor the
left/top, ceil the right/bottom, clip to the image. Require at least eight pixels
of corner separation and eight pixels of clipped width and height. Multiple or
missing faces and insufficient mouth support remain unassessed. No alternative
face, source or crop is selected based on detector response.

Use pinned Apache-2.0 DINOv2-small, singleton 112-square bicubic RGB inputs with
ImageNet mean/standard deviation. Concatenate the normalized final class token
and normalized mean final patch token: 768 float32 variables. Require every one
of the exact 223 float32 safe-tensor keys, strict loading and bitwise CPU tensor
values before moving to explicit deterministic MPS float32/eager attention.
Set CPU threads to one and MPS process fraction to 0.65; no fallback or download.
Model declarations, exact cached bytes and package versions enter the freeze.

Memoize only exact decoded RGB bytes within one source, with its fixed shape
and frozen models/procedure. Cache both availability and successful vectors;
record all actual face and image inputs and cache hits. Cache metadata, frame
indices, role, condition and known edit positions are not classifier variables.
This avoids repeatedly forwarding byte-identical pictures across the six
conditions; each picture still has its own native source clock and feature row.

For every native picture, independently route its timestamp to the existing
qualified mono16k waveform origin and cached five-state speech array. Require
an observed source sample and all ten retained ticks. Store the v122 3,840-variable
mean/standard-deviation summary only for supported positions. No new speech
encoder forward, extrapolated or zero-filled substitute is allowed. Keep exact
plans, origins and source clocks privately. These speech states can depend on
the whole padded input; their nominal tick interval is not finite local proof.

## Qualification and bounded execution

Before full extraction, qualify on the fixed first fitting source: the three
native centre-neighbour original pictures, the centre original again, and the
centre aligned construction. Verify exact audited bytes/RGB, require five
successful native mouth crops and five singleton image forwards. The repeated
original must have bitwise equal face coordinates/score, crop, normalized model
input and feature values. Construct one v122 2,304-variable visual descriptor
and one cached speech summary with exact native clock support. The constructed
feature distance is diagnostic only; it is not a detection result.

Freeze this code, tests and procedure before the first guarded stage. Reuse the
unchanged qualified owned-descendant monitor and v108 resource runner, outside
the sandbox. Run qualification, then eight exclusive six-source extraction
batches sequentially. Each neural stage retains the existing 900-second,
6-GiB sampled descendant RSS, 6-GiB initial RAM, 3-GiB initial project/archive
disk and 1.5-GiB running reserve limits. Admission failure creates no attempted
stage. A failed attempted stage is preserved without retry, overwrite, resource
relaxation or source/model substitution. Report samples as samples, not peaks.

## Interpretation and next comparison

All features, face coordinates, source identities, clips and individual media
fingerprints stay private on Seagate. Publish code, aggregate coverage and
whole-artifact/journal commitments only. Missing scope never means original.
No serving defaults, customer confidence, rollback model or payment change.
No Actions, outreach or deployment.

Later work must separately audit the feature/time routes, fit genuine-reference
projections and supervised visual-only, concatenation and interaction heads on
fitting groups, calibrate each using calibration groups, and compare locked
selection/confirmation outcomes with current checks. Do not let true masks or
edit boundaries choose candidate inputs. The centred single-generator bank also
requires placement and generator transfer testing before a general reliability
or replacement claim. Authored unit checks and encoder repeatability alone do
not establish manipulation detection, phonetic alignment or real-world accuracy.
