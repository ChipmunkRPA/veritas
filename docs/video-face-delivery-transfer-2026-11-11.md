# Same-source video delivery comparison

The earlier [whole-clip comparison](video-full-clip-face-swap-transfer.md) found different successful face-swap flags from the two frame checks. A subsequent cohort comparison suggested using the candidate on native HD and the reference on downscaled delivery. This test froze that routing rule before scoring and changed delivery on the same source pictures.

## Method

The first eight qualified `fusion_fit` groups in the frozen v95 order were selected after excluding both earlier whole-clip rosters. Each supplied one publisher original and two complete swaps from different generators. Source and donor-filename commitments were disjoint from both earlier rosters. Six groups used native 1080p pictures and two used 720p. The separate 24-group v95 confirmation reserve was preserved. The local source archive follows the [RTFS publisher rights and provenance review](video-common-source-preparation-v89.md), including the limits of donor attribution.

For each of 24 source clips, six fixed frame positions within the first six seconds were scored in both the original HD stream and an aspect-preserving 640×360 H.264 delivery. The delivery used Lanczos scaling, `libx264` CRF 23 and `yuv420p`. An initial preparation stopped before inference when the MP4 decoder exposed 149 of 150 intended pictures. The corrected preparation used the already verified native constant rational cadence. All 3,762 rendered picture timestamps matched the corresponding source prefix exactly. Each delivered case retained commitments to its six sampled images and the original source indices.

Both detector arms used the same decoded RGB, timestamp and face box for every sample within a delivery. Both ran the pinned, commercially admitted weights with float32 and one frame per forward. The four comparison points and normalized scales came from the earlier ten development originals and stayed frozen. The routing rule used the candidate median on native HD and the reference median on the 640×360 delivery. The paired maximum took the greater of the two normalized median responses and used its own development operating point.

## Local results

| Fixed method | Swaps flagged in native HD | Swaps flagged in 640×360 delivery | Original files flagged in both deliveries combined |
| --- | ---: | ---: | ---: |
| Reference median | 6/16 | 6/16 | 0/16 |
| Candidate median | 2/16 | 3/16 | 0/16 |
| Candidate top-two mean | 3/16 | 2/16 | 0/16 |
| Paired maximum of normalized medians | 5/16 | 6/16 | 0/16 |
| Proposed delivery routing | 2/16 | 6/16 | 0/16 |

All 48 inputs were assessed. The routing lost four native detections and added none against using the reference on both deliveries. The reference flagged the same number of swaps in each delivery, but changed its decision on six of the 16 matched swap pairs. Three gained a flag and three lost one. The 16 original deliveries represent eight source recordings. The sample cannot establish a general false-alarm rate.

The delivery change combines resizing and lossy re-encoding. The result therefore measures their joint effect in this panel. The proposed routing stays experimental and the serving defaults remain unchanged.

The paired verifier matched all 288 frame commitments between arms and independently recomputed the stored medians. GenD and the reference completed in 76.7 and 81.2 seconds. Their sampled process-tree peaks were 1.67 and 1.35 GB. The aggregate contains no media, facial imagery, identities, per-clip scores, or private paths. [Exact aggregate](evaluation-panels/video-face-delivery-transfer-2026-11-11.json)

## Checkpoint preflight for another detector arm

The GenD Perception Encoder checkpoint declares MIT, and the parent encoder's license is Apache-2.0. Its publisher reports similar mean video AUROC to the CLIP variant and different performance by benchmark. The pinned `4fd5a38cf071af2640d314110826ed5cd25cab88` checkpoint is 1,264,456,640 bytes with publisher SHA-256 `e2367f8db3a6b76f8d5cc5857f1ed0ea0993b33c9e1f428d3b06e5ca95de2213`. A bounded header read verified the 1024-to-2 head and 310 stored tensors. The upstream inference path uses 224-pixel inputs with dynamic image sizing and removes the encoder's projection head. The existing 336-pixel feature extractor is a different configuration. The full weights and native inference remain to be verified. [Pinned checkpoint card](https://huggingface.co/yermandy/GenD_PE_L/blob/4fd5a38cf071af2640d314110826ed5cd25cab88/README.md) · [Parent encoder license](https://github.com/facebookresearch/perception_models/blob/main/LICENSE.PE) · [Paper and per-benchmark results](https://arxiv.org/html/2508.06248v3)
