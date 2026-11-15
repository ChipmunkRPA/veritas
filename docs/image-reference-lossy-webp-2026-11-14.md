# Earlier-copy evidence for lossy WebP images

The local app can compare a lossy WebP upload with an earlier lossless
copy. The comparison shows a source crop from each image, the coordinates
of each measured region, and a mask the user can overlay on the WebP crop.
The downloadable comparison PDF and JSON include the same regions. A
highlight can come from ordinary editing and carries zero weight in the
AI assessment.

Enable the comparison with `VERITAS_IMAGE_REFERENCE_COMPARISON_ENABLED=true`
and `VERITAS_IMAGE_REFERENCE_LOSSY_WEBP=true` in development. It takes a
single opaque VP8 image in a simple WebP container, 512 to 1,024 pixels
per side, without changing source dimensions. The earlier file must be a
compatible 8-bit RGB PNG or lossless WebP. The two files need matching
framing. Animated, extended and transparent WebP containers are outside
this check. [WebP container specification](https://developers.google.com/speed/webp/docs/riff_container)
· [VP8 image header](https://datatracker.ietf.org/doc/html/rfc6386#section-9.1)

## Comparison method

The comparison first estimates whole-image alignment. It requires a stable
correlation peak within 0.75 source pixels of zero shift. It compares
Gaussian-smoothed brightness on the native grids and removes a common
channel offset fitted from distributed source samples. To establish shared
framing it requires at least 24 detailed 32 × 32 tiles, at least 65% of the
image’s detailed tiles, distributed across at least six of nine areas
covering every image third in both directions. The check requires at least
80% of pixels within the brightness allowance. A selected region exceeds
six smoothed brightness levels and covers at least 0.1% of the image after
removing small isolated responses. The crop always shows decoded source
pixels. The mask shows where the filtered comparison responded.

The first experiment tried re-encoding the reference at three WebP
settings. Its seven false highlights on unchanged files all came from
settings between the modeled qualities. Other files with different
encoding effort frequently went unassessed. A direct RGB residual check
highlighted 130 of 156 unchanged deliveries. The low-frequency brightness
comparison was selected using this 26-photo development bank. Commercial
model assets are unnecessary. Research on WebP compression sensitivity
motivated testing quality and effort independently. [Compression study](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Giakoumoglou_SAGI_Semantically_Aligned_ICCV_2025_supplemental.pdf)
· [Development photo and edit terms](image-edge-context-new-sources-2026-11-14.md#photo-qualification)
· [First experiment receipt](evaluation-panels/image-reference-lossy-webp-reconstruction-2026-11-14.json)
· [Filtered development receipt](evaluation-panels/image-reference-lossy-webp-development-2026-11-14.json)

The final rule correctly located an edit in all 312 staged edit
deliveries in the development bank. It left all 208 untouched and recompression-only
deliveries clear. All 156 same-image conventional copies produced
highlights. These real differences illustrate the scope of the evidence.

## Held comparison

The fixed comparison used 24 other source groups. Sixteen come from the
native regional confirmation bank, with at most one image per connected
group. Eight come from the rights-verified BG-REAL LaMa bank, with at most
one photo per recorded Flickr account. The roster excluded exact earlier
source hashes and close perceptual sketches across both banks. These
photos previously appeared in other project studies. The 648 WebP
deliveries use three new quality and encoding effort combinations. The
known mask comes from the exact changed pixels before final delivery. A
useful location must recover at least half those pixels and reach 0.30
intersection over union. [Native confirmation sources](native-regional-serving-v47-result.md)
· [LaMa photo and edit qualification](image-bg-real-transport-v225-result.md)

| WebP delivery | Useful sharp edits | Useful feathered edits | Untouched or recompression-only flags | Shifted or unrelated pairs declined |
| --- | ---: | ---: | ---: | ---: |
| Quality 65, effort 4 | 23/24 | 22/24 | 0/72 | 48/48 |
| Quality 82, effort 0 | 23/24 | 22/24 | 0/72 | 48/48 |
| Quality 92, effort 6 | 23/24 | 22/24 | 0/72 | 48/48 |

The earlier-copy route left all lossy WebP submissions unsupported before
this change. The new comparison found **135 of 144** useful edit locations.
All 144 edit deliveries produced regions, including nine with insufficient
overlap. Each profile found 31 of 32 native-bank deliveries and 14 of 16
LaMa deliveries. The 144 mild tone and blur controls in this held bank
produced no highlights. Strong conventional edits can still appear as
differences. This test did not estimate the accuracy of AI attribution or
the coverage of pure color changes with little brightness change. The
repeated deliveries share source photographs and do not represent 648
independent photos. [Held results and source commitments](evaluation-panels/image-reference-lossy-webp-holdout-2026-11-14.json)

## Local application verification

Three simulated USD 9.99 checkouts exercised a held edit, a recompression
and a translated file. No job started before payment confirmation. The
app found the held edit, showed no regions on recompression and returned
unassessed for the translation. On the edit, 11,783 of 11,892 marked
pixels were inside the known change. Source replay checked both crop byte
arrays and the exact mask in the export. The PDF names the same artifacts.
The comparison preserved each analysis result. [App verification receipt](evaluation-panels/image-reference-lossy-webp-app-2026-11-14.json)

## Local replay

The numeric receipts contain file hashes, the authored truth hashes and
measured mask hashes. They can be recounted with standard-library Python.
The native rerun recreates each delivered file and mask and checks the
recorded coordinates. The development bank is available from the pinned
publisher paths, with the acquisition command in the [photo study](image-edge-context-new-sources-2026-11-14.md#replay).
The held inputs use the retained native confirmation and LaMa source
bundles listed above. The outputs use libwebp 1.6.0.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
pilot=docs/evaluation-panels/image-reference-lossy-webp-reconstruction-2026-11-14.json
hold=docs/evaluation-panels/image-reference-lossy-webp-holdout-2026-11-14.json
python3 -B -S -m scripts.image_reference_lossy_webp_transfer review "$pilot"
python3 -B -S -m scripts.image_reference_webp_residual_holdout review "$hold"
backend/.venv/bin/python -B -m scripts.image_reference_webp_residual_holdout freeze "$run/freeze.json" \
  --development-sources "$photos" --prior-webp-results "$pilot"
backend/.venv/bin/python -B -m scripts.image_reference_webp_residual_holdout measure "$run/freeze.json" \
  --development-sources "$photos" --output "$run/holdout.json"
cmp "$hold" "$run/holdout.json"
backend/.venv/bin/python -B -m scripts.image_reference_lossy_webp_app "$run/serving" "$hold" --development-sources "$photos"
```
