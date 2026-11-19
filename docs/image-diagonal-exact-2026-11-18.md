# Exact diagonal copy evidence in lossless images

The added check can show two areas of one image whose original RGB pixels
match after reflection across either diagonal. The evidence shows one
verified pair with both source crops and their original pixel coordinates.
Other copies may be present. Ordinary editing and repeated graphics can
produce this relationship. It carries zero weight in the AI assessment.

The local option applies to upright, opaque, static PNG and lossless WebP
with each side from 512 to 672 pixels. Earlier copy checks keep precedence.
The diagonal check requires an exact rectangular match with texture in both
crops and different surrounding pixels. It can withhold a result when too
many equally plausible matches remain. The development option uses
`VERITAS_IMAGE_DIAGONAL_EXACT_COPY_REVIEW=true` with the existing
three-pair and affine checks. The shipped default is off.

## First comparison

Eight retained camera photographs supplied 24 authored single-copy cases
per carrier. Four additional patterned source photographs and controls from
the same camera sources supplied unchanged, conventional editing, earlier
copy transforms, and symmetry cases. The two encodings preserve identical
pixels and share the source groups.

The diagonal check added 18 useful locations out of 24 in each carrier,
across seven of eight camera groups. It added no misplaced pair and no
alerts on 56 ordinary or control deliveries per carrier. The first frozen
rule required no selected pair on eight separate files with two deliberate
diagonal copies. Three returned a pair, so the rule failed. Independent
pixel replay confirmed that all three returned pairs match an authored copy,
and that each file contains another copy. The complete replay reproduced
176 deliveries, 48 edit masks and 42 displayed RGB and perimeter witnesses.

## Separately frozen confirmation

The confirmation used six new camera source groups and four distinct
patterned sources from the retained bank. It set the requirement to verify
any displayed pair and disclose its scope before scoring. It separately
tracked untouched and ordinarily edited pictures, common repeated patterns,
whole-image symmetry, earlier copy detections, and multiple planted copies.

| Result per carrier | PNG | Lossless WebP |
| --- | ---: | ---: |
| Earlier check found a known diagonal edit | 0/18 | 0/18 |
| With the added check | 18/18 | 18/18 |
| Camera groups with an additional location | 6/6 | 6/6 |
| Added alerts on ordinary and earlier-transform controls | 0/50 | 0/50 |
| Misplaced additional pairs | 0 | 0 |
| Multiple-copy cases withheld | 6/6 | 6/6 |

The independent replay matched all 148 rendered delivery bytes and verified
the source geometry, RGB alignment, and perimeter difference for all 36
reported pairs. This is evidence for exact visible pixel copying at the
tested dimensions and encodings. The source groups and two encodings do
not establish a general-population error rate.
[Camera sources and attribution](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)

## Local application and portable counts

All eight isolated local analyses followed simulated checkout. Chromium
opened eight evidence pages and all four source crops. The PDF and JSON
matched each expected result, and all eight signed packages verified
against their originals. With the option enabled, single-copy PNG and
WebP fixtures showed the exact pair. Both kept the same AI assessment as
when the option was off. The untouched, whole-image symmetry, and
competing-copy cases showed no diagonal pair in this application test.
[Local application receipt](evaluation-panels/image-diagonal-exact-app-2026-11-18.json)

The [evaluation counts](evaluation-panels/image-diagonal-exact-2026-11-18.json.gz)
include sanitized delivery rows and evidence measurements. They contain no
media, source filenames, or crops. Recount the two panels with the
standard-library reviewer from the repository root.

```sh
python3 -B -I -S backend/scripts/review_image_diagonal_exact.py docs/evaluation-panels/image-diagonal-exact-2026-11-18.json.gz
```

## JPEG transfer trial

Two local panels tested the same diagonal search geometry using the existing
JPEG color comparison and its unchanged interior and surroundings checks.
Each panel encoded its images at two standard compression settings, JPEG
75 and JPEG 90. Both settings share the same photos and planted edits.
The candidate ran after the existing copy checks completed and left no pair.
It measured both crops on the decoded native pixels. A correct localization
required at least 50% overlap by intersection over union on each crop,
regardless of which crop the result listed first.

The first panel used six camera sources and four extra patterned controls.
The second used twelve distinct camera sources and six extra patterned
controls. The authored edits were new to these panels. The photographs came
from retained collections that were available to earlier image research.

| Measure | First, 75 | First, 90 | Second, 75 | Second, 90 |
| --- | ---: | ---: | ---: | ---: |
| Current check located a planted diagonal copy | 0/18 | 0/18 | 0/36 | 0/36 |
| With the candidate | 7/18 | 7/18 | 21/36 | 20/36 |
| Source groups with a new correct location | 3/6 | 3/6 | 8/12 | 8/12 |
| Added alerts on ordinary photos and ordinary edits | 0/22 | 0/22 | 0/36 | 0/36 |
| Added alerts on whole-image mirror controls | 0/10 | 0/10 | 2/18 | 2/18 |
| Added localizations below the required overlap | 0 | 0 | 1 | 1 |

The first study required eight new correct locations across at least four
source groups at each setting. The second showed a gain across eight groups
and missed the fixed control and location criteria. In both encodings, one
pair overlapped the intended feathered edit and its source with a smaller
intersection over union of 0.499. Two mirror controls at each setting yielded
matching crops that should have been recognized as part of a larger pattern.
Other transformation controls added no diagonal alerts. The original JPEG
candidate remained experimental after these two comparisons.

The [first panel](evaluation-panels/image-jpeg-diagonal-pilot-2026-11-18.json.gz)
and [second panel](evaluation-panels/image-jpeg-diagonal-confirmation-2026-11-18.json.gz)
contain numeric crop coordinates, comparisons and commitments. Recount the
reported locations and control alerts with the standard library.

```sh
python3 -B -I -S backend/scripts/review_image_jpeg_diagonal.py docs/evaluation-panels/image-jpeg-diagonal-pilot-2026-11-18.json.gz
python3 -B -I -S backend/scripts/review_image_jpeg_diagonal.py docs/evaluation-panels/image-jpeg-diagonal-confirmation-2026-11-18.json.gz
```

## Local JPEG comparison after the mirror test

The local JPEG option applies to upright color JPEGs with the standard
JPEG 75 tables and 4:2:0 color sampling. It compares the two original crops
after reversing one across a diagonal. The check withholds pairs whose
relative movement along that diagonal projects to less than eight pixels
in each coordinate. A near-axis match can be part of a larger mirror
pattern. Earlier copy checks keep precedence. The finding carries zero
weight in AI authorship confidence.

The separate test used original photos from ten different camera sources
and six distinct patterned groups, including four authored repeated layouts.
The test used 640 × 512, 544 × 608, and 704 × 512 pixels. The local option
covers a short side of at least 512 px and a long side of 608–704 px.
The reported error counts apply to the three measured sizes.

| Result | JPEG 75 | JPEG 90 |
| --- | ---: | ---: |
| Existing check located the authored displaced copy | 0/30 | 0/30 |
| With the diagonal check | 13/30 | 11/30 |
| Camera sources with a new correct location | 5/10 | 5/10 |
| Added ordinary alerts | 0/32 | 1/32 |
| Added whole-image mirror alerts | 0/32 | 0/32 |
| Added rotated-copy control alerts | 0/10 | 0/10 |
| New misplaced or partial locations | 0 | 0 |

The guard suppressed the single whole-mirror alert made without the guard
at JPEG 75. At each setting it also withheld five correctly located local
copies placed on a shared diagonal. This is a deliberate coverage limit
for this comparison. JPEG 75 passed the predefined evaluation criteria.
JPEG 90 fell one correct location short of the minimum of 12, and the
result for its one added ordinary alert was on an untouched photo.
The independent recount reproduced all 248 deliveries' schedule and paired
geometry. [Evaluation receipt](evaluation-panels/image-jpeg-diagonal-displacement-2026-11-18.json)

The isolated local application completed eight analyses after simulated
checkout. It rendered four original crop links in the browser and kept the
same image assessment for both examples with the option on and off. Every
PDF and JSON export matched the evidence on file, and the offline verifier
replayed all eight signed packages against the submitted JPEGs. Ordinary,
mirror, same-axis local, and other-compression controls produced no
diagonal pair. [Local application receipt](evaluation-panels/image-jpeg-diagonal-displacement-app-2026-11-18.json)
