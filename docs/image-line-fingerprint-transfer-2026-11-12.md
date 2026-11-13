# Row and column residual agreement around small image edits

This single-upload check found zero useful edit regions on 144 held deliveries. Its highest-ranked areas also missed all 144 before applying a cutoff. The check flagged 13 of 576 ordinary deliveries, including one unchanged photo and six processed views of that same photo in JPEG90. It remains offline. [Audited counts](evaluation-panels/image-line-fingerprint-transfer-2026-11-12.json)

## Frozen measurement

Camera processing can leave horizontal and vertical patterns. Goljan, Fridrich and Kirchner compared regional row and column power spectra in one image, and found that JPEG compression can suppress the signal. More recent work treats weak or conflicting local forensic cues as uncertain. This implementation tests a different, lighter measurement. It compares the signed residual at the same row or column coordinates across separated strips and requires the two reference strips to agree before using them. It uses no external code or trained weights. [Sensor linear-pattern study](https://ws2.binghamton.edu/fridrich/Research/Forgery-detection-using-LP-final.pdf) · [Evidence-guided localization study](https://arxiv.org/html/2607.29156v1)

Subtract a fixed Gaussian-smoothed luminance view, divide by local residual activity with a floor, clip residuals, and omit very dark or bright luma values. At each candidate rectangle, average its inner rows and columns and those of four 32-pixel flanks at the same native coordinates. A 16-pixel gap separates the rectangle and each flank. Subtract each of the eight recurring phase means from the line vector to reduce reliance on the final JPEG block grid. Both opposing flank pairs must reach a correlation of 0.16 and the line energy floor. The score is the weaker of the two drops between the flanks' correlation and the inner region's best correlation with a flank. It is a relative response with zero AI-confidence weight.

An authored 256 × 256 image with independently drawn row and column offsets verified the arithmetic before the holdout. The check found the exact 64 × 64 rectangle where both patterns were removed. The intact pattern scored below 0.001, and a constant image yielded no proposal. These synthetic cases validate implementation behavior.

## Source-separated evaluation

Eight ordinary calibration source groups each supplied the unchanged image, global processing, local autoencoder processing, blur, tone, sharpening, a conventional feathered same-photo copy and a whole-image resize round trip. Per format, the point is the second-highest source maximum or 0.08, whichever is greater. The points were fixed before scoring the held cases. Two source-separated held panels provide 24 photos split evenly between SDXL and Kandinsky edits. Recorded source, Flickr photo and photographer identifiers are disjoint across development and both held panels. The source photos had previous project exposure. Exact original and edited inputs were reconstructed from the verified publisher assets and processed on the native grid. [Earlier media, attribution and rights](image-background-patch-transfer-2026-11-11.md) · [Later source qualification](image-wavelet-region-transfer-2026-11-11.md)

Every source supplies a sharp generated insertion, its feathered counterpart, and eight ordinary controls. The controls include an unchanged source, local and global autoencoder processing, blur, local contrast, feathered processing, conventional local copying and resizing. All variants go through the same delivery recipe for a format. The JPEGs are directly encoded at quality 100 or 90 with 4:2:0 color. Truth uses the exact changed pixels before final compression. A useful rectangle needs 0.30 intersection over union and 0.50 edited-pixel recall. The comparator runs the current aligned JPEG luma check or PNG sharp-boundary check on the exact delivered bytes. Advancement required two new feathered locations per held panel and delivery, with limits on added ordinary flags and affected groups.

| Held delivery | Candidate useful sharp / feathered, out of 24 each | Comparator useful sharp / feathered, out of 24 each | Candidate / comparator ordinary flags, out of 192 |
| --- | ---: | ---: | ---: |
| PNG | 0 / 0 | 12 / 0 | 2 / 17 |
| JPEG100 | 0 / 0 | 1 / 1 | 1 / 3 |
| JPEG90 | 0 / 0 | 1 / 1 | 10 / 3 |

All 720 deliveries produced a measurable candidate rectangle. The candidate did not put a useful rectangle first in any of the three formats, so lowering the cutoff would leave all of these locations missed. All 13 candidate control flags were new relative to the comparator. Five occurred on resize controls, one on local contrast, and seven on a single JPEG90 source across its unchanged and conventionally processed views. The data supports a ranking failure on this method and these source-preserving edits. It does not measure customer prevalence.

## Local replay

The readback rebuilt all 720 JPEG or PNG deliveries and exact changed-pixel masks and checked 2,160 source-rectangle overlap counts. Only aggregate responses and commitments accompany the receipt. Source assets stay local. Set `development`, `earlier`, and `later` to the retained qualified intakes and `run` to a new path.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze fit confirm; do
  backend/.venv/bin/python -B -m scripts.image_line_fingerprint_transfer "$phase" \
    "$development" "$earlier" "$later" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_image_line_fingerprint_transfer \
  "$earlier" "$later" "$run" "$publication"
```
