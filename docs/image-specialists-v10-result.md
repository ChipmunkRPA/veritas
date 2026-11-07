# Regional image calibration: more edits found, more false alarms

The real-image specialist pilot completed all four frozen arms. Regional readouts
improved detection and probability loss over the pilot's whole-map summary,
but increased false alarms. Compression-specific calibration did not resolve the
weak JPEG signal. These results do not qualify a customer rollout; the current
PNG and JPEG checks, their rollback and overall confidence remain unchanged.

## Reserved-role results

These are 240 inputs from 60 source pairs: each genuine/edit image is processed
as PNG and JPEG quality 75. Each format has 60 genuine and 60 edited inputs.
All sources had already been consumed in earlier studies. Reassigning disjoint
calibration/fusion/test roles does not make this a fresh external benchmark.

| Frozen arm | Source/class-balanced file Brier | False AI /120 genuine | Missed AI /120 edited | Unassessed/tied /240 |
| --- | ---: | ---: | ---: | ---: |
| Whole image, pooled | 0.254825 | 10 | 98 | 0 |
| Whole image, format-specific | 0.253488 | 9 | 102 | 0 |
| Whole image + regions, pooled | 0.241323 | 23 | 74 | 0 |
| Whole image + regions, format-specific | 0.240569 | 20 | 76 | 0 |

The internal call counts use the frozen 0.5 tendency readout under a declared
50:50 reference prior. This is not a public threshold or validated upload
authenticity probability. The whole-image arm is a new research 95th-percentile
summary of the fixed map; it is not the complete current application or its
current connected-region display policy. The experimental TIDAL JPEG arm is not
the production JPEG detector.

The format-specific regional arm found 39/60 PNG edits, while falsely flagging
14/60 genuine PNGs. Its JPEG result found only 5/60 edits and falsely flagged
6/60 genuine inputs. Conversely, the whole-image format-specific JPEG arm made
zero false AI calls but missed all 60 edits. That zero alone is not a successful
check. Calibration cannot recover discrimination that is absent from its input.

Local region results also expose the tradeoff. In PNG, format-specific calibration
reduced falsely AI-leaning genuine quadrants from 73/344 to 56/344 relative to the
pooled regional arm, but missed 64/136 edited quadrants rather than 51/136.
These are coarse region associations, not assertions about individual pixel
authorship. Nearby genuine texture can affect a globally conditioned map.

## What actually ran and was verified

Use all 240 consumed source pairs from three edit families, 80 per family. The
frozen split supplies 60 groups to each of specialist calibration, combination
fitting, combination calibration and confirmation. Both original and edited
variants, both formats and all local regions stay in their source role. Every
head exceeded the unchanged 25-source-group/class support floor. Whole-image
and quadrant readouts share one dependence family; they are not five votes.

The preparation and separate source reviewer checked the actual image bytes,
167,586,843 source pixels, exact original/edit RGB differences and all 4,800
region-truth boxes. The reviewer independently reconstructed all 960 input byte
and decoded-RGB hashes, including the JPEG encodings. Ordinary global compression
changes were not relabeled as AI edits. Every location uses stored-source pixels.

Actual CPU inference completed 720 fitting/calibration inputs, then 240 reserved
inputs after all four artifacts were frozen. Sixteen first-cell repeat checks
were bitwise equal. All 144 available native PNG responses matched the older
publisher-runtime journal exactly (98 in training roles, 46 reserved), maximum
difference zero. The two guarded runs used about 1.06 GB maximum sampled tree
memory and completed without a failure or guard. Full response maps were not
duplicated; compact hashes and scalar summaries were retained.

The local model-free replay refitted all four arms and reproduced 3,840 file/local
probabilities exactly, including source coordinates and status bindings. Three
adverse checks preserved confirmation isolation, refused cross-role lineage and
withheld a combined score for a skipped required check. A separate Linux workflow
repeats scalar fitting only; it does not run the model or independently examine
source truth. Its actual receipt is recorded separately once observed.

## Reproduction and limits

The [protocol](image-specialists-v10-protocol.md),
[aggregate results](evaluation-panels/image-specialists-v10-result.json),
[source/condition audit](evaluation-panels/image-specialists-v10-source-condition-review.json)
and compact scalar packets are public. Attribute TGIF/TGIF2 to IDLabMedia and
preserve the [publisher's CC BY-SA 4.0 terms](https://github.com/IDLabMedia/tgif-dataset/tree/fc5c557bd1a2d49a5dabc3676a54cfc44798f4ea)
for the derived data packets as well. Project code retains its repository license.
Raw images, reference masks, map arrays and model weights are not published.

External pretrained-source overlap remains unknown. This consumed COCO-derived
inpainting pilot does not validate general image generation, face swaps, every
compression condition or arbitrary customer uploads. No outside human reviewer
participated. It supplies real evidence about the proposed calibration structure,
including a failure to improve edits at a comparable false-alarm rate. No source
reserve, public operating claim, serving calibration, billing or GCP resource was
changed. Passage and image development pilots are complete; video and noisy-voice
adapters and genuinely independent-source confirmation remain required.
