# Regional patch drift: completed development, rejected

The tested candidate did not improve image-edit evidence. With the frozen
genuine-only operating points, both readouts missed all48 known edits in each
format. The current selected component located12 in each. Keep the current
checks; no fresh confirmation, service integration or confidence change follows.
Zero highlights are not evidence that an image is original.

## Complete outcomes

All312 planned inputs completed:120 genuine-only calibration inputs and192
development inputs. Development contains48 original/edit pairs,16 per generation
family, in original PNG and whole-image JPEG75. These are consumed research
sources, not new held-out photos or customer uploads. Repeated formats are not
independent source samples. No completed input is omitted.

| Format / component | Located edits /48 | Missed edits /48 | Genuine photos highlighted /48 | Mean edit IoU |
| --- | ---: | ---: | ---: | ---: |
| PNG current | 12 | 36 | 2 | 0.183951 |
| PNG candidate, raw | 0 | 48 | 1 | 0 |
| PNG candidate, within-image | 0 | 48 | 0 | 0 |
| JPEG75 current | 12 | 36 | 3 | 0.103440 |
| JPEG75 candidate, raw | 0 | 48 | 1 | 0 |
| JPEG75 candidate, within-image | 0 | 48 | 0 | 0 |

Located means at least one selected pixel intersects the author-provided edit
mask; IoU additionally penalizes missed and wrongly highlighted pixels. An edit
can therefore be located poorly. Genuine highlights are regional false alarms,
not AI-file verdicts. The raw candidate's mean genuine highlighted area was
0.030504% in PNG and0.018427% in JPEG; the within-image arm highlighted none.

Current PNG located firefly5/16,sd2 4/16,flux-fill3/16; JPEG located2/16,6/16,4/16
respectively. Both candidate arms located0/16 in every family and format.
Current genuine flags by family were0,1,1 in PNG and1,0,2 in JPEG. The raw
candidate highlighted one sd2 genuine source in each format; within-image none.
The [complete review](evaluation-panels/image-v8-patch-drift-development-review.json)
retains family-specific IoU, areas, requirements and all denominators.

Each arm passed3/6 fixed requirements: completion, independent implementation
checks and the genuine-highlight budget. Each failed the minimum four added
located edits, nondecreasing mean IoU and no family overlap loss. No arm or format
advances. The operating points, readout, masks and rules were not retuned after
the negative result. No unused confirmation reserve was scored.

## What was tested and frozen

The [prospective protocol](image-patch-drift-v8-development-protocol.md) fixes
an adaptation of the MIT-licensed TRAIL perturbation method using an explicitly
Apache-2-licensed86.6M-parameter DINOv2 register-base backbone. It is a regional
inspection method with two same-image views, not a file-origin probability.
No localization head was trained. The backbone,448-square input,block10 readout,
0.2 Haar attenuation and map/display recipe were selected before project scores.
This is not the external paper's7B system or a reproduction of its accuracy.
See the [primary-source/native contract](image-patch-drift-v8-native-protocol.md).

The60 genuine calibration sources are disjoint from the48 development source
groups. For each format/readout, the fixed point is the second-largest reliable
genuine per-image maximum with strict `>` eligibility. All60 maxima were eligible
for each; exactly one calibration original exceeded each point. Edited photos did
not select points. The raw points are0.0758519172668457 PNG and0.07012814283370972
JPEG; the within-image points are40.344932556152344 and37.21767807006836.
These are inspection units, not percentages or promises of future false-alarm
rates. Points were published before edited-image inference.

The current-check comparator is an exact retained-scalar replay: selected TIDAL
PNG and full-image FUSED JPEG, preserving their original eligibility gates,
screening interpolation, reliability exclusions, quantization and component caps.
All192 source/input/array checks and current pixel metrics reproduced exactly.
This is not a new full-app baseline neural run. Both methods use the declared
OpenCV source-mask projection in this component comparison; actual production
packet admission, Pillow projection and total evidence caps are outside it.

The failure is practical: at a genuine-only limit suited to this trial, responses
did not produce edit-overlapping regions. Lower false alarms from empty masks do
not establish better detection. This result does not establish whether another
layer, backbone, point-selection method or perturbation would work. Those would
be new hypotheses requiring their own predeclared consumed-source development.
Pretraining overlap, source-image rights and performance on arbitrary customer
photos remain unestablished.

## Native and separate arithmetic checks

Before project scoring, all six authored synthetic images passed exact primary
numerical-reference and full-forward/block-hook checks; normalization and native
feature differences were zero. They establish implementation behavior only.

The separate full-data reviewer verified all312 source/tensor/map bindings,
median/readout calculations, reliability grids and eligible maxima. For every
development input, it rebuilt encoded grids, used a separate8-neighbour traversal
for component selection, and counted source pixels independently. All candidate
and baseline metrics match. Nearest-source projection additionally matched over
37,523,344 synthetic index pixels covering all55 source geometries.

All12 prospectively selected real-image probes reproduced block-hook features
exactly. Private feature-byte commitments passed; separately computed float64
cosine differs from the retained float32 map by at most3.3137083876866313e-7,
within the frozen5e-7 bound. This is implementation verification, not detection
quality or an independent person's review. Real photo features remain private.

## Operational failures retained

The first calibration launcher could not enumerate processes. It produced no
calibration output directory or prediction records, and no worker remained on
explicit inspection. The preserved failure precedes the successful same-code
120-input calibration. The second attempt's development run stopped at its
unchanged resource guard after34 successful inputs. A subsequent observation
found about1.61GB free disk and sufficient RAM; the generic guard receipt does
not identify the triggering resource contemporaneously.

Before repeating development, the
[public operational amendment](evaluation-panels/image-v8-patch-drift-operational-amendment.json)
specified the same192 inputs, unchanged code/model/points/masks/rules and guards,
reuse of byte-identical completed calibration, and exact repeat checks. It
records removal of585,634,832 bytes of recoverable duplicate scan files and the
877,599,816-byte downloaded cache of the unadopted Smaug research candidate.
Its public pinned download reference, license, source, original outputs and
all results remain; serving and rollback weights were retained.

The complete retry finished without guards. All34 repeated cases match exact
native input/feature/map commitments, both arm outcomes and the baseline.
Both earlier attempts remain privately retained. The final calibration took
47.859s, sampled RSS lower bound776,765,440 bytes; development78.132s and
807,829,504 bytes. Both have zero sampling errors. These are sampled process RSS,
not true peaks or full GPU-memory measurements. Counts refer to120 reused
calibration plus192 complete development cases, not346 independent inputs.

## Public reproduction and limits

The [scalar packet](evaluation-panels/image-v8-patch-drift-public-scalars.json)
publishes all312 scalar outcomes, source/response commitments and the local
audit. It includes no photos, reference masks, image embeddings, model weights
or customer files. Public source references retain their separate dataset terms;
the software license does not grant rights in those photos.

Run from the repository root with a clean Python standard-library environment:

```sh
PYTHONPATH=backend python -S -m scripts.reproduce_patch_drift_v8 review /tmp/patch-drift-review.json
PYTHONPATH=backend python -S -m scripts.reproduce_patch_drift_v8 tamper /tmp/patch-drift-tamper.json
```

Outputs refuse overwrite. The portable reviewer binds all312 roles/inputs,
refits four genuine-only points, recomputes576 development pixel-metric sets,
all family aggregates and all24 advancement checks. All13 deliberate alterations
are refused, including changed retry provenance. It is scalar-count review,
not new neural inference, source-mask verification or outside human testing.
The separately run local audit supplies the private pixel/vector checks above.
The new Linux workflow runs the scalar and alteration checks on a separate host;
an actual completed run receipt must be recorded before claiming it passed.

The current face-transition local release remains in place. Image checks,
customer conclusions, audio calibration, billing and GCP deployment are unchanged
by this experiment. Outside human reviewers remain zero. Improving regional
image edits, noisy/overlapping audio, lip sync, general video and writing remains
unfinished work in the broader improvement program.
