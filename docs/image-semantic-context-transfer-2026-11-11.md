# Comparing forensic responses with semantically matched image patches

The combined readout reduced ordinary highlights in PNG. All three arms found zero useful sharp or feathered regions across both test panels. The current regional routes stay in use. [Audited results](evaluation-panels/image-semantic-context-transfer-2026-11-11.json)

Recent work combines forensic and semantic image features for local edits. DEAR also studies how post-processing changes a detector's response. The DEAR publisher restricts the released weights to non-commercial use, so this study uses separately pinned MIT and Apache-2.0 checkpoints already available locally. This comparison is a separate implementation of an intra-image reference, with no publisher training code or restricted weights. [FUSED paper](https://arxiv.org/html/2608.28302v1) · [DEAR checkpoint terms](https://huggingface.co/k-aisi-anti-deepfake/dear-checkpoints) · [MIT forensic checkpoint](https://huggingface.co/OwensLab/commfor-model-384/tree/6076002bf0d9dd37537f965ee2f06f826c333b61) · [Apache-2.0 semantic checkpoint](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056)

Both encoders produce a source-mapped 24×24 patch grid. The reference arm uses the raw forensic projection. The semantic arm subtracts the median forensic response of the twelve most semantically similar patches more than five grid cells away. The joint arm takes the smaller of the two standardized responses. Each arm selects a 4-, 6-, or 8-cell square. The highest allowed review burden on eight separate development groups was one group with any flag. The points were sealed after 64 original, processed, blur and contrast views and before either comparison panel ran.

| Panel and delivery | Raw forensic highlights /72 controls | Semantic highlights /72 controls | Joint highlights /72 controls |
| --- | ---: | ---: | ---: |
| Earlier PNG | 11 | 4 | 2 |
| Earlier JPEG75 | 6 | 11 | 10 |
| Later PNG | 13 | 10 | 7 |
| Later JPEG75 | 6 | 7 | 9 |

The two panels contain 12 different source groups each, split evenly across SDXL and Kandinsky. Both were used by earlier project studies. Neither contributed to the current operating points. Each delivery has twelve sharp and twelve feathered edits in each panel. A useful location needs at least 0.30 intersection-over-union and 0.50 changed-pixel recall. The study verified 384 exact input bindings and 1,152 native-pixel measurements. The two encoders each ran 448 CPU forwards. [Earlier panel](image-boundary-blend-transfer-2026-11-11.md) · [Later panel and source checks](image-wavelet-region-transfer-2026-11-11.md) · [Local runner](../backend/scripts/image_semantic_context_transfer.py)
