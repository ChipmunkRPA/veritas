# Measured footprints for embedded JPEG preview differences

Chipmunk Verify now shows the exact marked cells beside the main image and embedded preview crops. The grid follows the aligned, trimmed preview and identifies where the comparison measured a difference. The two source crops retain their own coordinates. The footprint measures thumbnail-resolution evidence and carries zero AI authorship weight. [Original comparison](image-thumbnail-localization-2026-11-12.md)

## Compact region experiment

The experiment kept the current strict core and tried extending it through weaker nearby brightness or color differences corroborated by both resize methods. Each proposed extension had to be connected and compact, fit within 32% of each preview axis, cover at least half its own bounding box, and remain within four times the established core area. The candidate used 75% of the current residual threshold for these extensions. The comparison scored the actual source crop rectangles shown to customers, along with the underlying measured cells. A useful crop required 50% native changed-pixel recall and 0.30 intersection over union.

The freeze binds 222 files. Eight generated source groups include six previously unused sources from the SDXL and Kandinsky thumbnail comparison archives and two exact FLUX reference-edit composites from a separate construction. Ten independently credited Commons photos supply known conventional copy and feathered color edits. Each edit is delivered upright, mirrored with a stored-axis preview, and rotated with a display-axis preview. Routine variants include unchanged JPEGs, JPEG75, global tone adjustment and resize round trips. Thirty additional controls include ten untouched Pixel 6 Pro files and two ordinary variants each. The camera files have one credited contributor. All source terms and original-source commitments are in the ledger. [Alternate-generator qualification](image-wavelet-region-transfer-2026-11-11.md) · [Compact construction](image-flux-photo-preparation-v276-result.md) · [Commons sources](image-affine-native-scale-2026-11-11.md) · [Camera source](image-gainmap-localization-2026-11-12.md)

| Held outcome | Current crop | With extension |
| --- | ---: | ---: |
| Generated edits with useful crop | 18/24 | 19/24 |
| Conventional copy edits with useful crop | 24/30 | 24/30 |
| Conventional color edits with useful crop | 27/30 | 27/30 |
| Routine controls with any flag | 0/102 | 0/102 |
| Missing or unrelated previews accepted | 0/36 | 0/36 |

The extension added a useful location on one generated source and preserved all previously useful crops. Average generated crop recall rose from 74.7% to 77.7% and average intersection over union fell from 0.732 to 0.689. The requirement was three new source groups including two conventional groups. The candidate missed it and stays in the experiment. These are controlled stale-preview deliveries with earlier project exposure for other tasks. Unknown camera deliveries and the prevalence of stale previews remain outside the measurement. [Frozen case ledger](evaluation-panels/image-thumbnail-extent-transfer-2026-11-12.json)

## Local app verification

An independent native readback passed all 222 frozen input hashes. It compared 75 exact footprint PNGs using a separate connected-component implementation and checked 150 original crops against the source image and stored thumbnail pixels.

Three isolated Full analyses used simulated checkout. Two known edits using different EXIF preview conventions displayed an 18 × 7 footprint at (3, 55) on the aligned grid, with 80 and 83 marked cells respectively. The main and stored-preview crop bounds were verified independently. The API payload hashes matched the JSON records, and the extracted pixels matched the Standard and Summary PDFs. Each signed package verified against its source and included the exact Standard PDF. Chrome displayed all three views in the ordinary analysis page. The clean mirrored photo produced no preview-difference record.

## Replay

The public ledger reader uses the Python standard library. The native run also uses the exact retained source archives, which contain the credited photos and exact edit masks. Source and crop pixels stay local.

```sh
python3 -I -S backend/scripts/review_image_thumbnail_extent_transfer.py \
  docs/evaluation-panels/image-thumbnail-extent-transfer-2026-11-12.json

export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.image_thumbnail_extent_transfer freeze "$run" --wavelet "$wavelet"
backend/.venv/bin/python -B -m scripts.image_thumbnail_extent_transfer score "$run"
```
