# JPEG image localization comparison

The reserved comparison completed both JPEG qualities on the Mac. It used
59 photos in 48 source groups, with one known AI edit and four control views
per photo and quality. Both scoring paths completed all 1,180 planned
forwards and four exact repeats. Each mask was measured on the source
pixel grid. A useful location covered at least half the known edited pixels
and had at least 0.30 intersection over union with the edit.
[Protocol](image-jpeg-diagnostic-v189-protocol.md) · [Sealed aggregate](evaluation-panels/image-jpeg-diagnostic-complete-v189.json)

| JPEG quality | Check | Useful edit locations | Unedited flags | Tone-edit flags | Blur flags | Copy-move flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 75 | Current | 2 | 1 | 5 | 6 | 8 |
| 75 | Sensitive | 14 | 3 | 14 | 4 | 9 |
| 75 | Intersection of both | 1 | 0 | 0 | 0 | 2 |
| 90 | Current | 9 | 27 | 31 | 31 | 37 |
| 90 | Sensitive | 11 | 17 | 16 | 14 | 18 |
| 90 | Intersection of both | 3 | 6 | 5 | 4 | 5 |

Each table column has 59 photo views per row. The sensitive check gained
13 useful locations and lost one at quality 75. It gained seven and lost
five at quality 90. Its group-balanced gain was 18.1 percentage points at
quality 75, with a paired pointwise 95% interval of 6.9 to 29.5 points.
At quality 90 the gain was 3.1 points, with an interval of −9.4 to 16.0
points. Both qualities use the same 48 source groups.

The sensitive check flagged fewer controls of every type at quality 90.
At quality 75 its largest increase was nine more tone-adjusted photos
flagged. Requiring both checks to select the same pixels reduced control
flags and lost useful locations at both qualities. The sensitive check is
the stronger candidate for a separate JPEG trial. That trial should include
other editing families and realistic tone, blur and copy-move controls.

The trial covers one AI editing family. It measures the usefulness and
false highlights of regional evidence. JPEG90 false highlights remain
substantial for both individual checks. A customer-facing region would
describe a location for review and cite the observable evidence. Any
authorship estimate requires its own validation.
