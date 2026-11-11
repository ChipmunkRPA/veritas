# Finding two matching regions without mirroring

The image evidence view now checks for the same visual detail at two locations on the original pixel grid. It first groups nearby patch matches that agree on a displacement, then compares the aligned crops using local luminance correlation or exact pixel agreement. The existing geometric check runs when no dense match qualifies. When either check finds a pair, the app, PDF and JSON give the same source crops and positions. The evidence carries zero weight in the AI estimate. [Evidence builder](../backend/app/evidence/image_packets_v7.py)

Dense local matching and an alignment check can complement feature-point matching. The implementation extends Chipmunk Verify’s existing reflected patch method to translation, with the matching and corroboration settings frozen before the holdout. The paper code, weights and evaluation data were outside this comparison. [Dense-field matching](https://doi.org/10.1109/TIFS.2015.2455334) · [Alignment-based verification](https://doi.org/10.1155/int/9932351)

## Local comparison

On the 12-photo development set, the native patch method correctly located 9 copies in each format versus 2 with the geometric method. On a separate 16-photo set, each image supplied a 96 px and a 144 px square copy. The held-out roster excluded known creator and linked source groups from earlier copy and reference-edit panels. The separate hard-control set used eight previously qualified photos of repeated or symmetric detail. The images have recorded CC BY 2.0 terms. The fixed preparation also kept an original, a contrast change, and a local blur for each of the 16 source photos. Both formats share the same edit coordinates.

| Views | Geometric | Native patches | Native patches with geometric fallback |
| --- | ---: | ---: | ---: |
| PNG, 96 px useful | 4/16 | 13/16 | 13/16 |
| PNG, 144 px useful | 5/16 | 15/16 | 15/16 |
| JPEG90, 96 px useful | 3/16 | 9/16 | 10/16 |
| JPEG90, 144 px useful | 4/16 | 9/16 | 9/16 |
| PNG controls with any pair | 0/56 | 0/56 | 0/56 |
| JPEG90 controls with any pair | 0/56 | 0/56 | 0/56 |

A useful location requires each crop to reach at least 0.50 intersection-over-union with its authored box, in either order. Across the 64 edited views, the combined method added 31 useful locations across 13 photo groups and retained all 16 useful locations from the geometric method. One PNG and five JPEG edited views produced a pair below the overlap criterion. The 176 views are paired measurements from 24 source photos. They establish performance on these deliberately copied regions at the tested sizes and encoding. Stronger compression, edits mixed with scaling or rotation, and content copied from other images need their own comparisons. [Full aggregate](evaluation-panels/image-translation-patch-transfer-2026-11-11.json)

An isolated Quick analysis of an edit missed by the geometric method published the two matching crops, recovered the same native coordinates, and rendered the PDF. The model-free evidence path ran with learned inference disabled for that check. API and signed export tests checked the crop bytes, coordinate binding, and evidence from earlier versions. The comparison runner verifies the frozen input hashes and records all assessed and unavailable views. Run `python -m scripts.image_translation_patch_transfer --help` from the backend with the checked-out environment and the separately retained licensed source bindings. [Local comparison runner](../backend/scripts/image_translation_patch_transfer.py)
