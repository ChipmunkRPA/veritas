# Local spectral peaks in a single JPEG

This comparison tests whether a small edited area contains two compact
frequency peaks that stand out from the rest of the uploaded JPEG. Recent
research reported such peaks in some iteratively edited images and found
that camera imagery and JPEG compression can also produce them. Another
study found that whole-image classifiers can respond to changes outside
an inpainted area. The experiment uses single JPEGs with the original
surrounding pixels preserved before encoding. [Spectral artifact study](https://arxiv.org/html/2609.11317v1)
· [Inpainting exchange study](https://arxiv.org/html/2602.00192v2)

At 16-pixel steps, evaluate 40, 64, 96, and 128-pixel square windows at
the submitted resolution. Measure high-pass luminance with a Hann taper.
Within each spectrum, remove peaks close to the native 8-pixel JPEG grid
frequencies. Require two peaks at least 35 degrees apart and favor readings
that recur across adjacent windows but weaken in a larger neighborhood.
Select one rectangle over all four window sizes. The implementation uses
no pretrained weights and no code or assets from the two papers.

Fourteen previously prepared scene-text removal sources set the two
cutoffs. At each JPEG quality, take the second-highest score from the
per-source maxima on ordinary deliveries that the existing checks left
clear. A candidate must strictly exceed this cutoff. The freeze binds
the measurement code and all held manifests. Source-file and decoded-image
digests separate the calibration set from both held banks.

The first held bank contains 16 other scene-text sources with staged
Nano Banana 2 removals. Their reference images may have unknown origins.
The second contains 12 BG-REAL camera photos, each supplying LaMa and
MI-GAN removals. Edits in both banks use sharp and feathered masks. The
controls include unchanged files, local blur, tone adjustment, copying,
and conventional inpainting. Counts refer to delivered JPEGs. One source
can contribute several similar deliveries and both qualities. Known edit
masks identify the pre-encode changed pixels. A useful rectangle covers
at least half those pixels with at least 0.30 intersection over union.
[Scene-text publisher and terms](https://huggingface.co/datasets/QinmingZhou/TextEraseBench/blob/cf94f7eaffc654146cba02fe2f704cc1a6e4998d/README.md)
· [BG-REAL source and editor licenses](image-jpeg-frequency-bands-2026-11-17.md#held-photographs-and-test-controls)

## Local results

The calibrated points were 0.3460813 at quality 75 and 0.3268152 at
quality 90. The strongest rectangle missed the required overlap on all
56 calibration edits before either cutoff. Each point added flags on
five ordinary deliveries from one source image per JPEG quality.

| Held panel | JPEG quality | Useful locations, existing → combined | Ordinary flags, existing → combined | Misplaced candidate regions |
| --- | ---: | ---: | ---: | ---: |
| Scene-text removals | 75 | 0 → 0 of 32 | 10 → 20 of 80 | 4 |
| Scene-text removals | 90 | 0 → 0 of 32 | 0 → 15 of 80 | 6 |
| BG-REAL removals | 75 | 0 → 0 of 48 | 16 → 16 of 120 | 0 |
| BG-REAL removals | 90 | 4 → 4 of 48 | 10 → 10 of 120 | 0 |

The strongest new rectangle met the overlap rule on two of 160 held
edits before applying the cutoff. Both were sharp MI-GAN removals and
scored below the frozen points. On 138 of 160 edits, the new rectangle
had at least 0.50 intersection over union with the rectangle from the
same unchanged source. Five of the 25 added ordinary flags were on
unchanged scene-text deliveries. Each of the two held banks had prior
project exposure in other evaluations, and neither was used to set the
cutoffs. The candidate remains offline.

The existing comparison includes the aligned JPEG coefficient and
regional checks, with the serving repeat confirmation at quality 90
where applicable. A separate run of the application's earlier-grid
check produced no additional regions on the 280 held quality-90
deliveries. [Earlier-grid scope](image-previous-grid-2026-11-17.md)

An independent recount read back the encoded and mask hashes and
recomputed the exact rectangle intersections for all 756 calibration
and held deliveries. The run used NumPy 2.2.6, SciPy 1.18.1,
OpenCV 5.0.0, and Pillow 12.3.0.

[Calibration and locked source roster](evaluation-panels/image-local-spectral-peaks-freeze-2026-11-18.json.gz)
· [Held scene-text ledger](evaluation-panels/image-local-spectral-peaks-scene-2026-11-18.json.gz)
· [Held BG-REAL ledger](evaluation-panels/image-local-spectral-peaks-bg-2026-11-18.json.gz)

## Reproduce

Restore the earlier scene-text JPEGs, the held scene-text reserve, the
new BG-REAL JPEGs, and their conventional-fill controls using the prior
preparation steps. Keep their delivery manifests unchanged. The `freeze`
command reads the held manifests to bind source identities and file
commitments before any held image analysis. [Earlier scene-text inputs](image-jpeg-pscc-localization-2026-11-17.md#reproduce)
· [Held JPEG inputs and baseline replay](image-masked-reconstruction-2026-11-17.md#reproduce)

From `backend`:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=.venv/bin/python
module=scripts.image_local_spectral_peaks
"$runner" -B -m "$module" freeze "$development" "$development_baseline" "$freeze" \
  --held "$scene" --held "$bg" --held "$bg_fills"
"$runner" -B -m "$module" validate "$freeze" "$scene_result" \
  --banks "$scene" "$scene_baseline"
"$runner" -B -m "$module" validate "$freeze" "$bg_result" \
  --banks "$bg" "$bg_baseline" --banks "$bg_fills" "$bg_fills_baseline"
python3 -B -S -m "$module" review "$freeze" "$scene_result"
python3 -B -S -m "$module" review "$freeze" "$bg_result"
```
