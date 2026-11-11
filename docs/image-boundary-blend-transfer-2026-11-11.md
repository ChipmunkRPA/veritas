# Image localization after blending an edit boundary

The new boundary comparison localized five of 12 sharply inserted PNG edits and six of 12 JPEG75 edits. It missed all 12 corresponding edits in each format after feathering the border. The current regional routes also found zero useful locations on the feathered edits. The boundary method highlighted ordinary controls more often than the current routes, particularly in JPEG. It remains experimental. [Audited aggregate](evaluation-panels/image-boundary-blend-transfer-2026-11-11.json)

## Paired comparison

The study reuses the 12 source groups and publisher-derived SDXL and Kandinsky patches from the earlier image comparison. The additional inputs use a 12 px raised cosine ramp inside the patch rectangle. They include a feathered generated patch, a feathered autoencoder-processing control, and a same-site contrast adjustment. The six controls per group also include the original, existing local blur, and sharp local and global processing. Changes outside the known patch are zero before encoding. The edited-pixel masks come from exact native RGB differences after rounding the blend. The two deliveries share these precompression truth masks. The additional variants are new to local inference and the source photos were used in earlier comparisons. [Source rights, original patches and controls](image-background-patch-transfer-2026-11-11.md)

The boundary method compares native RGB steps with parallel steps nearby. PNG uses a 3 px displacement and JPEG uses 8 px so the comparison samples the same block phase in the local encoding. The method ranks rectangles using the geometric mean of four edge responses and tests three widths and three heights. It uses no learned weights. Before the 12-photo evaluation, the point in each format was set to the second-largest source maximum over eight separate original, processing, blur and contrast control groups. The chosen point applies strictly. The experiment tests whether apparent seam evidence survives smoother blending, motivated by the known sensitivity of local inpainting traces to image processing and compression. [Inpainting Exchange](https://arxiv.org/abs/2602.00192) · [JPEG inpainting study](https://pubmed.ncbi.nlm.nih.gov/42335053/)

| Format | Check | Sharp edits localized /12 | Feathered edits localized /12 | Control views highlighted /72 |
| --- | --- | ---: | ---: | ---: |
| PNG | Current regional route | 0 | 0 | 11 |
| PNG | Boundary prominence | 5 | 0 | 13 |
| JPEG75 | Current regional route | 1 | 0 | 7 |
| JPEG75 | Boundary prominence | 6 | 0 | 36 |

Useful localization requires at least 0.30 intersection-over-union and 0.50 recall of the changed pixels. Any reported region counts as a control highlight. Some views flagged by each method had little or zero overlap with the true edit. The 72 controls in each format are six views of the same 12 source photos, and the two formats reuse those photos. The table measures a controlled range of patch sizes, images and delivery conditions.

The comparison matched 120 prior current-route outputs by exact input SHA-256. It also made 72 new forwards, 36 per format, on the feathered edits and the two added controls. A separate readback reconstructed all 192 input digests and verified the source-pixel overlap arithmetic for 384 measurements. [Boundary runner](../backend/scripts/image_boundary_blend_transfer.py) · [Current-route runner](../backend/scripts/image_boundary_blend_reference.py)
