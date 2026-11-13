# Locate differences in a JPEG with a saved editor thumbnail

Chipmunk Verify can now compare an upright JPEG with a supported thumbnail carried in its editor image resources. When the two views align, the ordinary results page and PDF show the main image crop, stored thumbnail crop and measured difference footprint. The JSON export records the exact crop positions, the preview carrier and the PNG commitments. This detects differences between two views stored in the file. The comparison gives zero weight to AI authorship confidence.

The accepted carrier is a single RGB JPEG thumbnail in the standardized APP13 image resource 1036. The parser checks the resource length, thumbnail dimensions, scan format and orientation. Malformed, repeated or legacy competing resources are declined. Existing usable EXIF thumbnails retain priority. The additional path requires an upright main image without an EXIF thumbnail entry and follows the existing signed-provenance stop. The same alignment, background-agreement and two-resize residual tests apply. [Image resource format](https://www.adobe.com/devnet-apps/photoshop/fileformatashtml/) · [Prior EXIF localization](image-thumbnail-localization-2026-11-12.md)

## Frozen carrier comparison

The study moved the exact saved preview JPEG bytes from EXIF into the tested editor carrier while preserving the main JPEG scan. All 166 upright cases came from the prior exact-mask panel. They include controlled feathered edits from three generation methods, conventional copying and local color adjustments, routine re-encodes, camera files and incompatible previews. All 166 comparisons reproduced their previous EXIF status, boxes and mask measurements. Two independently credited original Wikimedia files supplied real editor thumbnail bytes. Removing their EXIF previews isolated the editor resource and both files measured no region. These tests measure transport compatibility and localization on a controlled archive. They give no estimate of how frequently editors retain a stale thumbnail or how this check performs on unselected uploads. [Prior source and mask commitments](image-thumbnail-extent-transfer-2026-11-12.md) · [Source credits and result](evaluation-panels/image-editor-preview-2026-11-12.json)

| Outcome | Result |
| --- | ---: |
| Known generated edits localized | 6/8 |
| Conventional local copies localized | 8/10 |
| Conventional local color changes localized | 9/10 |
| Flags on routine views | 0/102, with 100 comparable |
| Flags on missing or foreign previews | 0/36 |
| Flags on real editor-thumbnail controls | 0/2 |
| Rejected malformed or ambiguous carriers | 6/6 |

A useful region covered at least half the exact changed pixels and had at least 0.30 intersection over union with the change mask. Two generated edits had highlighted areas that did not reach the localization criterion. Retouching can produce measurable differences, and the thumbnail may have unknown origin or age.

## Local verification

An isolated Full analysis worker accepted a known generated edit and an ordinary control after simulated prepaid checkout. The positive returned two source crops and one footprint. The actual API assets, JSON bounds and both Standard and Summary PDF crops agreed, including decoded image pixels. The control produced no thumbnail finding. The worker used the actual preview implementation while the unrelated model registry lanes were disabled for the scoped test. The signed package included the verified Standard PDF. The thumbnail PNGs use the separate export/source check below.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for step in freeze run review; do
  python -B -m scripts.image_editor_preview_transfer "$step" "$archive" "$public_sources" "$run" || exit
done
python -B -m scripts.review_image_editor_preview "$original" "$export_json" "$crops"
```

The carrier replay uses the exact previous thumbnail extent archive and the two pinned Wikimedia originals. For source readback, save the API crop PNGs from the export under their SHA-256 filenames in `$crops`. The verifier reopens both images, recomputes the footprint from the submitted bytes, and checks every recorded position, asset commitment and decoded pixel. It rejected altered source, position and crop bytes in the focused tests.
