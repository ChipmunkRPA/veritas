# JPEG localization struggled on LaMa object removal

Both checks finished all 480 planned local forwards and four exact repeats. The comparison used 24 qualified photo pairs with known AI-edited pixels, including 19 publisher test sources and five validation sources. Each pair contributed the same five views at JPEG 75 and 90. The seal verified all 20 scoring phases and ten pixel readouts. [Method](image-bg-real-jpeg-v229-method.md) · [Verified aggregate](evaluation-panels/image-bg-real-jpeg-complete-v229.json)

| JPEG quality | Check | Useful AI edit locations | Unedited flags | Tone-edit flags | Blur flags | Copy-move flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 75 | Current | 1 | 0 | 0 | 0 | 7 |
| 75 | Sensitive | 0 | 4 | 4 | 3 | 10 |
| 75 | Intersection | 0 | 0 | 0 | 0 | 2 |
| 90 | Current | 1 | 2 | 2 | 4 | 11 |
| 90 | Sensitive | 0 | 6 | 7 | 5 | 19 |
| 90 | Intersection | 0 | 0 | 0 | 1 | 4 |

Each count is out of 24 source views in its column. A useful region covered at least half the known edited pixels and reached at least 0.30 intersection over union. The sensitive check flagged more of every conventional control at both JPEG qualities. The intersection reduced copy-move flags and returned empty regions for all AI edits. This differs from the earlier JPEG 75 result, where the sensitive check found 14 useful locations and the current check found two. [Earlier editing-family comparison](image-jpeg-diagnostic-v189-result.md)

The sensitive check remains a development candidate. The two benchmarks show that its localization benefit depends on editing family. The same 24 source groups recur at both JPEG qualities. These measurements apply to pixel evidence, and no calibration or serving change followed.
