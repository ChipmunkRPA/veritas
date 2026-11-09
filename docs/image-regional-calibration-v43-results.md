# Reserved calibration rejected the standalone regional candidate

On these calibration images, the adapted candidate localized more native edits
than the current check, but **none of the 28 frozen settings qualified in either format**. Every
setting raised the original-image false-alarm count above the current check.
The candidate is not admitted to the app or overall confidence. The point,
diagnostic and confirmation sources remain unforwarded.

## Actual execution

The [exact published execution freeze](evaluation-panels/image-regional-calibration-v42-execution-freeze.json)
preceded the first reserved prediction. All 28 producer phases completed:
39 photos in 32 source components, AI/original/tone/blur views, two formats and
both arms. This is 624 unique model calls plus four prescribed exact repeats.
All repeats passed. All 14 map-readout phases and the final seal passed, with
no guard, sampling error or recorded resource crossing. The readout made no
neural calls. All four assets, the final learned checkpoint and 47 frozen
implementation/protocol bindings were verified again after execution.

The [complete aggregate result](evaluation-panels/image-regional-calibration-v43-results.json)
retains every policy, including all rejected ones. The
[execution receipt](evaluation-panels/image-regional-calibration-v43-execution.json)
binds terminal resource-profile hashes and the sealed result. Raw maps, source
identities, pairings and learned weights remain private.

## Localization and false alarms

Useful localization requires both IoU at least 0.30 and edited-region recall
at least 0.50. The useful fraction below first averages photos within each
source component, then averages the 32 components equally. False alarms count
components with any retained region in that control kind, out of 32.

| Format and readout | Useful localization, component mean | Original false alarms | Tone false alarms | Blur false alarms |
| --- | ---: | ---: | ---: | ---: |
| Native current check | 20.8% | 3/32 | 7/32 | 11/32 |
| Native candidate, strictest frozen setting | 66.7% | 6/32 | 7/32 | 5/32 |
| JPEG75 current check | 4.2% | 1/32 | 2/32 | 1/32 |
| JPEG75 candidate, strictest frozen setting | 6.3% | 4/32 | 11/32 | 3/32 |

The candidate rows illustrate the fixed grid endpoint, raw response above
0.90 with at least 1% source-space area. They are **rejected descriptive
examples, not selected policies** or customer confidence thresholds. Across
all 28 candidate settings, native original-image false alarms ranged from
6 to 20 components; JPEG75 original-image false alarms ranged from 4 to 17.
Thus neither format had an eligible setting under the frozen requirement of
no observed increase separately in original, tone and blur controls.

At the individual-photo level, the current check usefully localized 8/39
native edits and 2/39 JPEG75 edits. It retained no region on 28/39 and 31/39,
respectively. Its native original/tone/blur false flags were 3/39, 7/39 and
11/39; JPEG75 counts were 1/39, 2/39 and 1/39. These photos are correlated
within source components. There is no selected candidate outcome count;
that absence must not be represented as zero detections or zero errors.

Every planned measurement was available: each arm/format had 156 valid
source views, with zero failed, skipped, unavailable, inconclusive or missing
map measurements. That is processing coverage, not a claim that authenticity
was conclusively determined. A clear or absent region does not establish
whole-file originality.

## Inference and next work

The native candidate has useful region-placement information, but its false
alarms prevent standalone replacement. JPEG compression sharply reduces its
useful localization, and tone controls remain a particular weakness. The
current checks also miss many edits. Keeping the current checks is a rollback
decision, not evidence that their reliability is adequate.

Next, evaluate a separately frozen composite that uses the candidate only to
refine regions on files already flagged by the current check. This can be
developed on these now-observed calibration maps; it is not a fresh independent
test. Any selected composite must then face the untouched source groups before
adoption. The original failed standalone comparison remains unchanged.

This cohort covers one source-relative generative editing family and does not
establish detection of wholly generated images, unseen generators, face swaps
or real-world prevalence. Source components approximate independence; creator
aliases and pretraining overlap remain limitations. These are internal
localization settings, not calibrated authorship probabilities. No serving
default, public threshold claim, protected audio calibration, workflow,
external outreach, billing or GCP resource changed.
