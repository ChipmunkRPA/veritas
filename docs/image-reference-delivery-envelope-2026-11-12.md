# Paired image localization beyond ordinary delivery differences

This test compares an earlier JPEG with a cropped and resized PNG. Existing registration must identify the earlier native crop before the new measurement evaluates local color differences. A strong difference can come from conventional retouching or a generated edit. The measurement carries zero weight in the AI authorship assessment. [Earlier cross-format comparison](image-reference-cross-format-local-edits-2026-11-12.md)

Recent work found that image classifiers can lose much of their ability to recognize inpainting when original pixels are preserved outside the edit. Reference-based research compares corresponding areas to collect more specific evidence. Our comparison uses RGB pixel measurements and no outside model weights or implementation. [Inpainting Exchange](https://arxiv.org/html/2602.00192v2) · [Reference-based photographic assessment](https://arxiv.org/abs/2611.10908)

## Frozen measurement

The earlier copy is an upright standard quality-95 RGB JPEG with 4:2:0 chroma sampling. The submitted file is an upright RGB PNG. The existing correlation-based crop comparison searches for a proportional resize and rejects unsupported alignment. Four resize kernels and two versions of the earlier pixels, including an ordinary quality-83 recompression, estimate a per-pixel range of expected color. Each version receives robust whole-image tone adjustment. The new mask highlights coherent color changes outside that expected range, with a wider allowance in areas where the simulated deliveries disagree. A Gaussian filter reduces the influence of fine texture and compression noise.

The 144 development views include six SDXL and six Kandinsky source groups. Development selected one of eight prespecified settings and sealed it before the 120 held pairs ran. The held inputs include six separately credited FLUX edit groups and six credited public photo groups. All recorded contributors are distinct across development and held. Held photos have prior project exposure, and the same FLUX deliveries were used in the preceding comparison. The new setting uses only the separate development sources. Tests use exact masks of changed pixels on the submitted grid and independently check each accepted earlier crop against all four native crop edges. A useful result must cover at least half the known changed pixels and reach at least 0.30 intersection over union. [Source rights and masks](image-reference-cross-format-local-edits-2026-11-12.md) · [FLUX composition](image-flux-photo-expansion-v285-result.md) · [Wikimedia photo credits](image-jpeg-public-replay-2026-11-12.md)

Development selected a Gaussian sigma of 1.6 pixels and a six-level minimum RGB excess. It localized 10 of 12 sharp and 10 of 12 feathered generated edits and 11 of 12 ordinary local color changes. It highlighted zero nuisance views and accepted no incorrect crop on development.

## Held results

| Outcome | Current app | Unguarded crop comparison | Fixed 10-level comparison | Local delivery range |
| --- | ---: | ---: | ---: | ---: |
| Useful sharp FLUX locations | 0/6 | 4/6 | 2/6 | 3/6 |
| Useful feathered FLUX locations | 0/6 | 3/6 | 0/6 | 2/6 |
| Useful ordinary local color changes | 0/12 | 10/12 | 10/12 | 10/12 |
| Useful ordinary local blur | 0/12 | 0/12 | 0/12 | 0/12 |
| Highlights on ordinary delivery controls | 0/36 | 5/36 | 0/36 | 1/36 |
| Incompatible pairs accepted | 0/48 | 0/48 | 0/48 | 0/48 |

The one delivery nuisance was a global contrast adjustment with 1,269 selected pixels. None of the 24 untouched or recompressed views highlighted. One ordinary local blur delivery highlighted too few pixels to meet the overlap criterion. On the three generated overlays below criterion, all selected pixels fell within the true edited areas. Those overlays covered 682 of 1,843, 670 of 6,155, and 639 of 4,793 changed pixels. Two other source pairs were unassessed at crop registration for both sharp and feathered edits. All accepted crops had the correct native geometry.

The comparison passed the held requirements for sharp edits, conventional color edits, delivery nuisances and crop accuracy. It fell short of the prespecified three useful feathered locations and produced three generated highlights below the overlap criterion, one over the limit. The first held panel therefore did not support enabling this check on its own.

## Separate inpainting challenge

Eight distinct BG-REAL LaMa photo groups provide 64 more pairs. Each original, generator edit and publisher mask is bound to the retained qualification, and every native truth mask is verified against exact changed RGB pixels. The fixed selection excludes overlapping recorded photo and contributor identities. The sources have earlier project exposure. For each group, evaluate the sharp and feathered edit, the ordinary crop, earlier recompression, global contrast, local blur, a fractional shift, and a different photo. The point and geometry rules remain fixed. The additional gate requires four useful feathered locations, at most one nuisance highlight, and zero wrong or incompatible crops. [BG-REAL source qualification](image-bg-real-transport-v225-result.md) · [Source rights](image-bg-real-roster-v218-method.md)

| Outcome | Current app | Unguarded crop comparison | Fixed 10-level comparison | Local delivery range |
| --- | ---: | ---: | ---: | ---: |
| Useful sharp LaMa locations | 0/8 | 7/8 | 7/8 | 7/8 |
| Useful feathered LaMa locations | 0/8 | 7/8 | 7/8 | 7/8 |
| Useful ordinary local blur | 0/8 | 1/8 | 0/8 | 0/8 |
| Highlights on ordinary delivery controls | 0/24 | 1/24 | 1/24 | 1/24 |
| Incompatible pairs accepted | 0/16 | 0/16 | 0/16 | 0/16 |

The same global contrast adjustment caused a highlight under all three experimental methods. The new method selected 6,892 pixels in that control. It highlighted zero untouched, recompressed or local blur views. One LaMa source was unassessed on both edited inputs. The separate challenge passed all of its prespecified checks. Across both held banks, the new method added two useful feathered locations and one sharp location over the fixed 10-level comparison. It highlighted two ordinary delivery controls out of 60, compared with one under the fixed method and six under the unguarded method. Both new nuisance highlights came from global contrast adjustments.

## Local app

The authenticated **Compare with an earlier copy** feature offers this check when existing comparisons are unassessed. Enable `VERITAS_IMAGE_REFERENCE_JPEG_PNG_ENVELOPE=true` alongside `VERITAS_IMAGE_REFERENCE_COMPARISON_ENABLED=true` in local development. The comparison shows the earlier area used, both native crops, and the difference overlay. Its separate comparison PDF and JSON record the same measurements. The offline JSON reviewer repeats the original routing decision and measurement, then checks source hashes, geometry, masks, overlays and crop pixels. This check is a separate comparison between two supplied copies. It carries no authorship weight.

An isolated worker completed three Full analyses after simulated prepaid checkout. The held feathered edit selected 10,244 pixels, with 10,243 inside the 11,194-pixel truth mask. A recompression of that source selected zero pixels. The global contrast nuisance produced the expected 6,892 selected pixels. The API, PDF and standalone JSON replay agreed on all three. Each unpaid event queued zero analyses. Chrome displayed three evidence images and the interactive overlay, downloaded the PDF and exported matching JSON. [Public receipts](evaluation-panels/image-reference-delivery-envelope-2026-11-12.json)

The serving comparison also reproduced admission, all four crop edges, and the exact native mask digest on every one of the 184 held pairs. It admitted 57 of 120 FLUX and ordinary pairs and 46 of 64 LaMa pairs. [Serving replay counts](evaluation-panels/image-reference-delivery-envelope-2026-11-12.json)

## Reproduce

Use the retained first and later publisher intakes from the previous cross-format study, the pinned FLUX archive, and the eight-file public JPEG replay cache. Set `earlier`, `development`, `public_photos`, and `run` to the two publisher intakes, the public cache, and a new scratch directory respectively. The native replay reconstructs the sources and masks and checks rendered crops and overlays. The standalone reader independently checks source commitments, localization arithmetic, the development selection, and the held gate. The public receipt omits media, per-file source identities, and private paths.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for stage in freeze develop held review; do
  backend/.venv/bin/python -B -m scripts.image_reference_delivery_envelope \
    "$stage" "$earlier" "$development" "$public_photos" "$run" || exit 1
done
python3 -B -I -S backend/scripts/review_image_reference_delivery_envelope.py "$run"
```

Set `lama` to the retained BG-REAL v225 qualification and `lama_run` to a new scratch directory. Run the same sealed development point on the additional family and repeat the native measurement and export audit.

```sh
for stage in freeze run review; do
  backend/.venv/bin/python -B -m scripts.image_reference_delivery_envelope_lama \
    "$stage" "$run" "$development" "$public_photos" "$lama" "$lama_run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_image_reference_export "$export" "$earlier_jpeg" "$submitted_png"
```
