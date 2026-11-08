# Video face evidence calibration: completed development comparison

Keep the existing video checks. Adding appearance evidence improves this study's
calibrated crop-only reference, but the existing raw appearance cue is stronger
than every new combination. A better result against a weak reference is not a
reason to replace a stronger available check or spend a fresh reserve.

This is a fixed-support, single-visible-face experiment on consumed development
sources, not the full application's video verdict, identity proof, lip-sync test,
fresh benchmark or customer accuracy estimate. No serving default changed.

## Source-bound experiment

All 648 exact prepared clips completed four source-disjoint native stages:
164 specialist-calibration, 164 combination-fitting, 160 combination-calibration
and 160 reserved development clips. The 162 known source components each contain
an original, an ordinary edit and two inserted face swaps. The source audit
decoded 93,043 RGB frames and verified 12,420 changed AI source frames inside
their declared intervals. All 14,256 local units were fixed before scores.
Actual frame clocks are used, including older 25 fps ordinary controls.

GenD crop responses, SFace appearance transitions and DINOv2 patch-feature changes
have separate local heads and a separately fitted/calibrated combination. The
declared weights have commercial-admitted MIT/Apache-2.0 records. All heads,
four combinations and internal evaluation points were fixed before reserved
authenticity inference. The same 39 completely assessed genuine calibration
components supplied a budget of three false-flag components for every arm.
This internal comparison is not a public service threshold or promised rate.

## Reserved development results

Each row has 80 manipulated and 80 genuine clips. Missed AI includes inconclusive
clips. A file false flag and a source component with any false flag are different
denominators; each component has two genuine control files.

| Method | AI found /80 | Missed AI /80 | Genuine false flags /80 | False-flag components /40 | Inconclusive /160 | Source/class-balanced Brier |
|---|---:|---:|---:|---:|---:|---:|
| Calibrated crop |11|69|5|4|9|0.243463|
| Crop + appearance |29|51|5|4|9|0.201888|
| Crop + feature change |11|69|5|4|9|0.243324|
| All three |29|51|5|4|9|0.201978|
| Existing raw crop cue on these supports |72|8|70|35|9|—|
| Existing raw appearance cue on these supports |62|18|1|1|5|—|
| Existing raw feature-change cue on these supports |8|72|1|1|5|—|

Raw cue decisions are binary inspection flags, not calibrated probabilities; no
Brier is invented for them. Their points are the existing code constants, not
values selected after these results. These rows use the same fixed input schedule
and cue arithmetic; they are not the complete production tracker, adaptive window
search or application's fused result. Do not present the raw crop false-alarm
count as the whole service's false-alarm rate.

Appearance-containing combinations pass the **predeclared crop-reference
continuation gate**, retained as `qualifies_for_fresh_source_followup:true` in the
unaltered result. That narrow gate does not establish superiority to the stronger
raw appearance reference. Both combinations lose 33 AI detections, add four
genuine false flags and have four more inconclusive files relative to that cue.
The release decision is therefore **no promotion and no fresh-reserve test of
this combination yet**. The original protocol eligibility is not rewritten.

## Localization and evidence scope

Flagged support spans separated by at most 550 ms are merged using the frozen
rule. Best merged-span IoU with the known edit must be at least .3. Bridged gaps
are interval estimates: only the explicitly listed source frames were examined.
Frame presence and source timestamps are not face/pixel-authorship proof.

| Method | Edited clips localized /80 | Genuine clips with a flagged span /80 |
|---|---:|---:|
| Calibrated crop |1|5|
| Crop + appearance |63|7|
| Crop + feature change |16|9|
| All three |63|11|
| Existing raw appearance cue |62|1|

Local flags use their separately frozen local points, so they need not equal
file flags. One extra localized clip with six or ten additional genuine clips
highlighted does not justify replacing the existing appearance cue. Pixel masks,
face identity and audio/lip-sync conclusions were not tested.

## Native verification and reproducibility

All four native roles and guarded fitting/review completed with exit 0 and no
resource guard. Native role durations were 380.12, 393.32, 375.24 and 365.99 seconds.
Sampled process-tree RSS lower bounds were approximately 1.80–2.89 GiB; these
are not complete accelerator-memory or peak-memory measurements. Short-lived
decoder child sampling races remain recorded.

The native inputs comprise 22,032 face-detection, 7,662 crop-classifier, 21,765
appearance and 21,765 patch-feature inputs, plus 12 explicit repeat inputs.
Every SFace alignment matched its independent transcription bitwise. The frozen
crop/patch repeat limits and SFace exact repeats passed. The independent local
binding reviewer checked all 648 source/frame/time commitments, local labels,
counters and explicit missing observations. No exactly equal eligible crop input
crossed source roles. There are 114 unavailable crop units and 75 unavailable
units in each temporal view; they remain unassessed. This review re-executes no
neural model and is not outside human review.

The first v19 engineering attempt failed before its first detector forward due
to a wrapper/payload handoff. Its original code, freeze and failed receipt remain
public. The explicit corrected v20 attempt uses the same sources, roles, labels,
weights and mathematics and adds a wrapper-contract regression.

The [public scalar release](evaluation-panels/video-face-specialists-v20-release.json)
contains source-bound local time/frame positions, scalar measurements, fitted
heads and combined probabilities. It contains no clips, crops, landmarks, face
features, alignments, per-frame hashes or customer uploads. Public identity
remapping reproduced private probabilities exactly. Local scalar refit reproduced
all 11,168 assessed probabilities exactly. Adverse checks changed 3,416 reserved
measurements without changing fitting, preserved 104 unassessed measurements,
withheld a combined score when a required check was removed, and rejected
cross-role lineage. The local immutable-commit replay archived 53 exact declared
files and reproduced the same probabilities with maximum difference 0.

The actual [Linux run](https://github.com/ChipmunkRPA/veritas/actions/runs/42432887072)
completed successfully on immutable science commit
`89c80d8f31857935db62dfa08005d87aae7140c2`. It archived the same 53 files,
refitted all four arms, compared 11,168 probabilities with maximum difference
2.220446049250313e-16, and passed all three adverse checks. The
[actual receipt](evaluation-panels/video-face-specialists-v20-linux-review.json)
records run/job identity and the unaltered replay output. An initial dispatch
before GitHub registered the new workflow returned 404 and started no job;
the verified workflow then ran once. This success does not resolve the earlier
passage fitter convergence failures or constitute outside human review.

Next development should compare an appearance-only calibrated reference and a
jointly fitted combination that retains each view's signal. Family maximum
pooling loses which check supplied the maximum; the observed tradeoff motivates
testing this limitation, but does not prove it is the sole cause. Include stronger
genuine edits, actual person changes, and missing-view controls before new
fresh-source/full-app verification. Preserve old results and serving rollback.
