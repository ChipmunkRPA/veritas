# Learned face responses on AI mouth edits

Decision: **reject the candidate**. The source-audited v5.1 comparison localized
0 of 16 known AI mouth edits, the same as the current sequence baseline. It
failed the unchanged improvement and spatial-overlap gates. No fresh actors
were scored, no new customer detector was enabled and overall confidence is
unchanged. No outside reviewer has participated.

## Scope and method

The [frozen development design](mouth-artifact-v5-development.md) uses all 48
consumed v4 cases from eight CREMA-D actors: 16 neural mouth edits and 32
untouched/ordinary-edit/timing controls. GenD's existing trained forgery head
replaces generic feature change as the signal; there is no new head training.
Source centres, triplets and tracked context match the production sequence
check. Native aligned crops use float32 GenD preprocessing, rather than the
baseline's reduced 112-pixel DINO input.

The candidate measures a middle-window increase in fake-minus-real logit against
both flanks, discounting disagreement between the flanks. This is a local
full-face response, not a mouth mask or authenticity probability. Its point
uses non-AI development maxima only. Zero non-AI flags on those same inputs do
not establish a future false-alarm rate.

The pinned [publisher checkpoint](https://huggingface.co/yermandy/GenD_CLIP_L_14/tree/891ce014a0308386c4d7d25b3dcf436a22db5504)
declares MIT, as do its source and base CLIP encoder. The installed weight SHA is
`d76f0bdfd74a29fe1b1c1b84a80ac92486993e426878e8c7a3944281fbb96833`.
Permitted production YuNet replaces the restricted upstream face detector.
Source data retains [CREMA-D ODbL/DbCL terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt).
Publisher declarations do not establish complete training-data rights or
pretraining independence. No new checkpoint was fetched for this comparison.

## Invalid first attempt and correction

The source audit invalidated v5.0: its experimental runner associated captured
native crops with sorted feature keys. FrameStore emits cached frames before
uncached frames, so those orders differed. All 48 cases failed their first
native-pixel comparison. Those localization numbers are invalid and are not
used to judge the detector's quality.

The old source is retained at Git revision `549c89e`; local raw receipts and the
[invalid-attempt notice](evaluation-panels/mouth-v5-invalid-first-attempt.json)
remain separate. V5.1 joins crops to insertion keys before sorting and checks
the exact reduced crop at that boundary. A regression exercises the actual
FrameStore cache behavior. Window arithmetic, point selection, measurement
region and advancement gates were verified AST-identical to v5.0. Only the
binding and implementation pins changed; no actor/case was substituted.

## Complete valid outcomes

| Condition | Planned | Baseline localized AI | Candidate localized AI | Baseline / candidate non-AI flags |
|---|---:|---:|---:|---:|
| AI redraw, same-frame target | 8 | 0 | 0 | — |
| AI redraw, earlier target | 8 | 0 | 0 | — |
| Untouched control | 8 | — | — | 0 / 0 |
| Local brightening | 8 | — | — | 0 / 0 |
| Non-neural crop/resizing | 8 | — | — | 0 / 0 |
| Constant audio delay | 8 | — | — | 0 / 0 |

All 48 executions were assessed without failures, with exact v4 baseline
window/centre replay. The candidate point is 1.7028230202694736 logit-contrast
units. The largest AI-redraw response was 0.307713 and largest retimed-mouth
response 0.422630; ordinary crop/resizing reached 1.702823. This rule reacts
more strongly to some ordinary resampling than to these known mouth edits.
Lowering the point after seeing results would be a new development design.

No localizations were gained or lost; mean temporal and spatial IoU over all AI
cases are zero. Coverage/control gates passed; the required three extra
locations and 0.01 mean spatial-overlap improvement did not. The full
[comparison](mouth-artifact-v5-evaluation.json) preserves all 48 outcomes. These
results concern this contrast check, not the entire app or every use of GenD.

## Source correspondence and external arithmetic review

Independent local replay checked **2,016 model inputs across all 48 clips**:
source-file hashes, native decoded RGB, exact PTS, freshly detected YuNet
alignment, native crop pixels and float32 preprocessed tensor hashes matched.
The [source audit](evaluation-panels/mouth-v5-source-audit.json) passed all cases.
This proves input correspondence, not outside review or predictive accuracy.

The [public responses](evaluation-panels/mouth-v5-public-responses.json) include
allowlisted source identities, input hashes/logits, reported windows, baseline
windows, exact-clock references and frozen code/weight identities. Media,
private paths and model binaries are omitted. The standard-library
`review_mouth_artifact.py` checks reported-window arithmetic, timestamp metadata,
control-point selection and complete temporal denominators, without ML packages.
It does not certify classifier execution, actual pixels, missing eligible
windows, track membership or spatial masks. Full replay needs separately
acquired media/weights. See the [review instructions](evaluation-panels/README.md).

The standard-library replay exactly matched the valid temporal counts. All 32
focused tests and lint/whitespace checks passed. Inference used local MPS with
a CPU baseline; GCP/CUDA parity is untested.

- V5.1 lock: `baf33bae9751a945681e566e3848593e95860efbb22efae7fca5169b8657e11a`.
- Score receipt: `57f42b591105455d2bf25dadcb37c555d8be69b06191992acfb0c9d9ea2ce5cd`.
- Source audit: `3b1eaaf132a2dc62aebcb84bfdf59be173105051948c10dd5137fea3bf23353a`.
- Public responses: `a54e474404ccc8f136306bafeb4844e51dd86ad31b62df5a97f29146430c6339`.

## Next experiment

Do not promote this check or consume fresh actors to rescue it. A direct
mouth-artifact representation needs separately specified source-group training,
ordinary-edit negatives and calibration. Remaining consumed actors can support
development. A subsequent frozen confirmation needs unseen actors, sentence
families and a second permitted editing family. Learned visual artifact
detection and audiovisual correspondence need separate comparisons. Current
reports should continue treating audio/picture timing as context.
