# Three exact copy pairs in one lossless image

Chipmunk Verify can locate three separate pairs of matching areas in a
supported lossless image. The evidence view shows both original crops for each pair,
how they align, and where their original coordinates sit in the image. The
PDF and JSON carry the same six crops and pixel measurements. The signed
package can verify those measurements against the submitted image.

The optional local check covers upright, opaque 512 × 512 RGB PNG and
lossless WebP stills. The evaluated combinations include a same-orientation
copy and a vertical reflection, plus either a horizontal reflection or a
180-degree turn. Every pair must have sufficient texture, identical pixels
after alignment, and differences on at least three surrounding sides.
Regions in different pairs must be at least 16 pixels apart. A fourth
qualifying pair or conflicting geometry can prevent the three-pair result.
Other areas can fall outside the search. Exact copies, intentional symmetry
and natural repetition can produce matches. The comparison carries zero
weight in AI authorship confidence.

Set `VERITAS_IMAGE_THREE_EXACT_COPY_REVIEW=true` with
`VERITAS_IMAGE_DISJOINT_EXACT_COPY_REVIEW=true` to enable this in local
development. The new option defaults to false. The existing two-pair proof
and its historical signed reports remain verifiable.

## Held comparison

The frozen panel used eight distinct attributed CC BY 2.0 camera images
from the qualified BG-REAL archive and six additional Commons photographs
with repeated or symmetric detail. The two-pair development bank used
different sources. Source groups may have appeared in other project studies.
All source files, edited deliveries, source geometry and comparison rules
were fixed before scoring. The PNG and WebP cases use the same images and
are paired delivery tests. The two layouts use conventional copy-and-paste
edits, so the results measure localization of repeated pixels.
[BG-REAL source qualification](image-bg-real-transport-v225-result.md) ·
[Commons acquisition and terms](image-affine-native-scale-2026-11-11.md)

| Measure | PNG | Lossless WebP |
| --- | ---: | ---: |
| Files with all three copy pairs, previous → combined | 0/16 → 12/16 | 0/16 → 12/16 |
| Useful individual pairs, previous → combined | 20/48 → 44/48 | 20/48 → 44/48 |
| Source groups with an additional useful pair | 6/8 | 6/8 |
| Incorrect new pairs on three-edit files | 0 | 0 |
| Ordinary files with added three-pair evidence | 0/22 | 0/22 |
| Shared endpoints with added three-pair evidence | 0/8 | 0/8 |
| Nearby endpoints with added three-pair evidence | 0/8 | 0/8 |
| Four individually qualifying pairs correctly withheld | 7/7 | 7/7 |

The broader experimental search displayed three qualifying pairs in the
remaining four-edit file. Those pairs were two reflections and one half-turn.
The app withholds that combination because the evaluated release combinations
each include a same-orientation match. The production source reader replayed
all 140 uploads and retained every one of the 24 correct three-pair results.
Fourteen of 16 three-edit files had three individually qualifying rectangles
per format. Twelve returned every authored box exactly. The gate required at least eight improved files across three
source groups per format, no added ordinary alerts or wrong pairs, and no
new three-pair evidence on shared or nearby endpoints or fully qualifying
four-pair cases. Both formats passed. Original-source replay verified all
140 encoded inputs and the measured boxes and perimeter differences for
every returned pair.

## Recount the held results

The [portable receipt](evaluation-panels/image-three-exact-copy-pairs-2026-11-18.json)
records anonymous groups, integer crop geometry, transform families and
paired comparison counts. It contains no images or crop pixels. The
standard-library reviewer independently checks the pair separations and
recounts exact locations and controls.

```sh
python3 -B -I -S backend/scripts/review_image_three_exact_pairs.py \
  docs/evaluation-panels/image-three-exact-copy-pairs-2026-11-18.json
```

With a signed package, the matching verification key, and the original
upload, the existing [original-file reviewer](independent-testing.md#verify-image-crops-from-a-signed-export)
checks the six unannotated crops and all three source measurements.

## Local app verification

Ten simulated USD 9.99 checkouts completed without creating an analysis job
before payment. The two positive uploads each displayed three exact pairs
and six original crop positions with the option on. With the option off,
neither displayed three exact pairs. The automated assessment matched before
and after enabling the option. The exact two-pair fixtures still displayed
two pairs. Untouched and fully qualifying four-pair fixtures displayed no
new exact three-pair evidence.

Chromium followed all 20 original source links across the active positive
and legacy cases. API, PDF, JSON and signed original-file replay agreed on
all eight active cases and the two feature-off positives. A translation can
share identical crop pixels and therefore share a stored PDF image while
appearing in two distinct source positions.
[Local application receipt](evaluation-panels/image-three-exact-copy-pairs-app-2026-11-18.json)
