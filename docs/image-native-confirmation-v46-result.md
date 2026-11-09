# Native regional confirmation supports a substantial sensitivity gain

The previously selected sensitive readout usefully localizes **45/59 edits
versus 10/59** for the current native check. It adds 36 useful locations and
loses one. Original false flags rise from **3/59 to 13/59**. This is a large
detection gain with a real false-alarm cost, not an automatic failure under
the former per-control zero-increase rule.

The selected agreement-only mask usefully localizes **9/59**, loses one current
useful location, and flags **0/59 originals, 1/59 tone controls and 0/59 blur
controls**. It is a narrower complementary view, not an equally sensitive
detector or independent proof of AI origin.

## Fixed confirmation and complete outcomes

Both readouts, the trained checkpoint and the current comparator were selected
from the completed v45 comparison and bound before confirmation. The
[protocol](image-native-confirmation-v46-protocol.md),
[exact freeze](evaluation-panels/image-native-confirmation-v46-freeze.json)
and successful preparation receipt were published and remotely verified before
the first confirmation prediction. No model, calibration, source, setting or
component rule was refitted on these labels.

All **59 source photos in 48 connected components** supplied an AI edit,
original, tone and blur view. These known source groups were held out from
project fitting/calibration/comparison. They share the earlier editing family;
unknown creator aliases, pretrained overlap and original-photo authenticity
remain limitations. This is not a new-generator or population guarantee.

Each column below contains 59 correlated photograph views, not 59 independent
people. Useful localization requires both source-mask IoU >=0.30 and edit
recall >=0.50. Empty masks stay in the denominator.

| Readout | Useful edit locations | Edited photos with no region | Original false flags | Tone false flags | Blur false flags |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current native check | 10 | 44 | 3 | 3 | 12 |
| Selected sensitive readout | **45** | **6** | 13 | 12 | 9 |
| Selected agreement-only view | 9 | 44 | **0** | **1** | **0** |

The extra sensitive regions merit integration and further scope testing. The
false originals are material: roughly 22.0% of original-photo controls were
flagged here, versus 5.1% current. They must not be hidden behind the larger
detection gain or the lower combined any-control false-group count.

## Group-balanced results and uncertainty

Average useful localization across the 48 components rises **20.8% to 75.0%**.
The paired gain is **54.2 percentage points**, with conditional paired-group
bootstrap 95% interval **[39.6, 68.8] percentage points**. Group-mean IoU rises
from 0.1918 to 0.5293. These are localization measurements, not authorship
probabilities or customer confidence percentages.

Original/tone/blur false groups out of 48 are **3/3/12 current**, **11/10/7
sensitive** and **0/1/0 agreement-only**. Any-control false groups are 14, 12
and 1. The sensitive original false-group difference interval is approximately
[4.2, 29.2] percentage points; tone is [4.2, 27.1]. The increases are genuine
measured costs under this conditional comparison, not merely a reporting error.

Agreement-only useful fraction is 18.75%, with paired difference -2.08 percentage
points and interval [-6.25, 0.0]. It should be described as trading sensitivity
for fewer review regions, not outperforming every aspect of current.

Average original-photo area wrongly highlighted is **0.154% current versus
0.745% sensitive**. Tone changes from 0.175% to 0.715%, and blur from 1.545%
to 0.609%. Agreement-only has zero original/blur wrong area and 0.000481%
tone wrong area. The complete aggregate retains all counts and area measures.

Intervals use 10,000 fixed-seed paired component draws and linear percentiles.
They hold the fitted models/calibration and selected data fixed; they exclude
their uncertainty and broader source/generator shifts. They are pointwise,
not simultaneous across readouts and not court reliability estimates.

## Execution and reproducibility

All **474 neural calls** completed: 472 unique views across two paths and two
exact first-input repeats. All 20 producer phases, ten model-free readout
phases and the model-free final summary exited zero without a guard, sampling
error or recorded resource crossing. This is not a complete second neural replay.

Raw float32 maps passed save/readback and source-byte/RGB/geometry/truth binding.
All 472 standalone mask measurements matched the frozen production readout
before 236 actual intersection masks were measured. Separate contingency and
group arithmetic reproduced each method's useful counts, false-photo/group
counts and source-balanced fractions. Both used native assets, all 64 code
bindings and the completed checkpoint were verified again. The protected
audio calibration is unchanged.

The [full aggregate](evaluation-panels/image-native-confirmation-v46-result.json)
and [execution receipt](evaluation-panels/image-native-confirmation-v46-execution.json)
bind every outcome and terminal resource-profile hash. Forty-nine focused local
checks and Ruff pass. Resource limits remained the original 900 seconds/6 GiB
sampled worker-tree RSS per phase and RAM/disk reserves. RSS is sampled, not
a guaranteed peak or a complete GPU-memory measurement.

No raw media, source identities/pairings, maps, feature values or learned
weights are published. Outside human reviewers remain zero. No workflow,
outreach, real charge or GCP resource was used.

## Local integration follows, without an authorship-confidence shortcut

Keep the sensitive and shared-region views as distinct evidence behaviors.
Do not multiply their responses as independent probabilities. Native/PNG
results do not qualify a JPEG replacement. This study does not establish
AI-specificity against ordinary copy/paste or spatial retouching; such controls
need a separately specified test. Regional evidence can support inspection
without proving AI origin or contributing to overall authorship confidence.

A [compact private adapter](evaluation-panels/native-regional-inference-artifact-v1.json)
has been exported for the production loader: 24,062,644 bytes, 246 tensors and
6,009,089 parameters. Every tensor roundtrips exactly to the accepted state;
optimizer state is omitted. The artifact is local and not published. This
does not yet establish production-forward parity or serving admission.

Next, implement a reentrant production loader and adapter, verify exact native
geometry/masks, then check the normal view/crops, model-name filtering, PDF,
JSON and paid-before-analysis path before a new local default. Existing
models/rollback, overall confidence, USD9.99 test-mode prepayment and SaaS
limitations remain unchanged. Broader multi-model writing, audio and video
work remains part of the active objective.
