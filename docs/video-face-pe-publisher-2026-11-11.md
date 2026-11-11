# Paired face-swap review on an outside publisher and ordinary camera edits

The same paired review used in Chipmunk Verify was scored on an outside publisher's face swap and ordinary edits on eight genuine camera recordings. The 10 original development clips had already fixed both solo points and the paired point. This study left those points, the six-frame sampler and the minimum of four usable samples per arm fixed. [Development comparison](video-face-pe-transfer-2026-11-11.md) · [24-group prospective comparison](video-face-pe-confirmation-2026-11-11.md)

## Sources and method

- **FaceSwapAI.** The public source is an unswapped synthetic still-state sequence and the publisher's full-length production face swap. Both adult identities were made synthetically for the study, which the publisher released under CC BY 4.0. The matched deliveries include an unchanged reference, a routine recompression and four color interpretations of the same face swap inserted from 3.000 up to 6.000 seconds. All four variants represent one swap from one source group. The exact decoded RGB outside the interval matches the unchanged delivery. The insert can also contain color or background differences. [Publisher study](https://faceswapai.com/resources/video-face-swap-continuity-study) · [Versioned manifest](https://faceswapai.com/video-continuity-study-v1.json) · [Local input preparation](video-faceswapai-matched-v186-method.md)
- **Microsoft VCD2.** Eight genuine camera recordings from eight distinct camera types supplied three matched deliveries apiece. Each triplet contains unchanged, chroma-converted and recompressed views. The last two changed pictures only from 3.000 up to 6.000 seconds. The VCD2 data is distributed under CDLA-Permissive 2.0. Participant overlap is unknown. [Microsoft dataset and rights](https://github.com/microsoft/VCD#dataset-license) · [Local control qualification](video-vcd2-matched-controls-v290-result.md)

The frozen native sampler picks six interior pictures over each full clip. Both arms checked the same source bytes, presentation timestamps, RGB pixels and largest face boxes. Each arm took the median of its usable face responses. The paired rule takes the larger of the two normalized medians, with at least four responses required from each. The tested media came from outside the model-point development and prior paired confirmation source groups. Earlier local studies used some of the media with other checks. Both arms scored every delivery with the pinned decoder and checkpoints on a local Mac.

## Outside publisher results

| Delivery | Reference flags | Complement flags | Paired flags |
| --- | ---: | ---: | ---: |
| Full face swap | 0/1 | 0/1 | 0/1 |
| Partial face swap | 0/4 | 0/4 | 0/4 |
| Full and matched unswapped synthetic reference | 0/2 | 0/2 | 0/2 |
| Matched synthetic recompression | 0/1 | 0/1 | 0/1 |

All eight deliveries yielded six usable responses per arm. For the partial swap, the sampler checked 1.417 and 2.833 seconds before the edit, 4.250 and 5.708 seconds inside it, and 7.125 and 8.542 seconds after it. Both arms' responses on the four flanking samples matched the unchanged delivery exactly. The complement's raw response at 5.708 seconds increased on all four variants. Its response at 4.250 seconds decreased on all four. The paired score was identical to the unchanged delivery. The unchanged synthetic reference also produced an elevated response at 5.708 seconds, so a high isolated response would need its own evaluation before it could support localization.

## Genuine camera controls

| Control | Reference false flags | Complement false flags | Paired false flags |
| --- | ---: | ---: | ---: |
| Unchanged | 0/8 | 1/8 | 1/8 |
| Chroma-converted | 0/8 | 1/8 | 1/8 |
| Recompressed | 0/8 | 0/8 | 0/8 |
| All deliveries | 0/24 | 2/24 | 2/24 |
| Recording groups with any flag | 0/8 | 1/8 | 1/8 |

Both flagged deliveries came from one recording. Its unchanged and chroma-converted views crossed the paired review point, and its recompressed view stayed below it.

All 32 videos had six usable responses per arm. A separate readback independently verified all 192 native frame indices, presentation timestamps, RGB hashes and matched face boxes per arm, and reproduced the reductions and counts. The local Metal runs took 505.951 and 508.387 seconds, including full-file verification, video decoding and model loading. Peak sampled process-tree memory was 1.05 and 1.54 GiB. [Aggregate and source receipt digests](evaluation-panels/video-face-pe-publisher-2026-11-11.json)

## Scope

The publisher clip tests a distinct production swap pipeline on one synthetic source. Its four partial deliveries test sensitivity to decoding assumptions on the same edit. Eight genuine camera recordings measure ordinary-processing flags across this local control panel. Generalization to a broader set of outside generators and localization of short edits require additional source groups. The paired face check remains an opt-in review of sampled faces. Its result carries zero overall confidence weight.

The proposed VCF and DeepSpeak source collections have noncommercial terms. The publisher's VCF data access terms apply to the videos separately from the open-access paper. Those collections were outside this evaluation. [VCF access terms](https://docs.google.com/forms/d/e/1FAIpQLSckwg4UgdaYL1v3HIbaRY9mSFntIFnfQwF17SJFHqnaMi4iFw/viewform) · [DeepSpeak paper](https://arxiv.org/abs/2408.05366)
