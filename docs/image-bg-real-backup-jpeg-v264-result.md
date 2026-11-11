# JPEG localization on twelve additional object-removal pairs

Both checks completed 240 unique local model calls and four exact repeats on 120 matched JPEG inputs. The 12 source groups supplied the same five views at each quality. The seal verified the preparation, 12 producer phases, and six pixel readouts. [Frozen method](image-bg-real-backup-jpeg-v264-method.md) · [Exact aggregate](evaluation-panels/image-bg-real-backup-jpeg-v264-complete.json)

| JPEG quality | Check | Useful edit locations | Unedited flags | Tone flags | Blur flags | Copy-move flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 75 | Current | 0 | 1 | 1 | 1 | 1 |
| 75 | Sensitive | 0 | 0 | 0 | 0 | 5 |
| 75 | Intersection | 0 | 0 | 0 | 0 | 0 |
| 90 | Current | 0 | 2 | 2 | 3 | 4 |
| 90 | Sensitive | 0 | 4 | 4 | 3 | 8 |
| 90 | Intersection | 0 | 0 | 0 | 0 | 0 |

Each column has 12 views per row. At quality 90, the current check touched the known edit in two images, with too little overlap to count as useful. The sensitive check flagged three edited images at quality 90, and its highlighted areas missed the known edited pixels in all three. The shared mask produced no regions in this set. This confirms that the tested sensitive check and its intersection offered no useful improvement on these additional LaMa removal edits. [Earlier LaMa comparison](image-bg-real-jpeg-v229-result.md)

The first preparation monitor stopped before launching a worker because its subprocess wrapper omitted the stderr redirect. The corrected preparation reproduced the expected plan and froze the exact inputs before inference. The original startup receipt was retained privately. The comparison covers pixel evidence from one editing family and two JPEG settings.
