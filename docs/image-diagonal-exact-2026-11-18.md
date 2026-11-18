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
