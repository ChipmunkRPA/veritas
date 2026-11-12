# Image edit localization with features from three encoder layers

A three-layer patch check localized zero of 96 edited deliveries. A last-layer-only check localized one feathered JPEG75 edit and flagged 12 of 72 ordinary JPEG75 deliveries in that panel. These results keep both candidates outside the app. [Audited counts and frozen method](evaluation-panels/image-multidepth-transfer-2026-11-11.json)

## Comparison

DINO-FIL reports that combining early and late encoder features can help a larger, partially tuned model localize inpainting. We tested a smaller version of that idea with an existing Apache-2.0 DINOv2-small checkpoint and a separately implemented linear fit. Each image maps to a 24 × 24 grid. One arm uses the final layer. The other combines normalized patch features from layers 3, 6, and 12. The checkpoint stays frozen. [DINO-FIL](https://arxiv.org/abs/2610.21780) · [Pinned encoder and terms](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056)

The fit compares the known edit interior and distant patches within eight matched Stable Diffusion 1.5 photo groups. Each source contributes equally. Each sharp and feathered edit is compared with five variants of the same source, including the original and ordinary processing. Twelve other rights-qualified Commons photos supply ordinary calibration controls. Those photos come from the larger-image copy study and are chosen in digest order after checking the intake and its recorded attribution. The point is strictly above the second-highest control-source maximum per arm and delivery. [Paired fit sources](image-bg-real-insertions-2026-11-11.md) · [Commons control sources](image-affine-native-scale-2026-11-11.md)

The two held panels use 12 photos each, with separate recorded Flickr accounts and photo IDs from fitting and from one another. Each includes six SDXL and six Kandinsky source groups. The panels had earlier project exposure. Commons-to-Flickr identity overlap and encoder training exposure are unverified. The same sharp and feathered truth masks serve PNG and JPEG75, with six ordinary controls per photo. The baseline reruns the existing regional PNG boundary and JPEG double-quantization measurements on the exact same deliveries. [First held panel](image-background-patch-transfer-2026-11-11.md) · [Second held panel](image-wavelet-region-transfer-2026-11-11.md)

## Results

Each row contains 12 sharp edits, 12 feathered edits, and 72 controls. A useful region has at least 0.30 intersection over union and 0.50 changed-pixel recall. Each cell gives sharp localizations, feathered localizations, and ordinary highlights.

| Panel and format | Existing check | Final layer | Three layers |
| --- | ---: | ---: | ---: |
| Earlier PNG | 5 / 0 / 13 | 0 / 0 / 0 | 0 / 0 / 6 |
| Earlier JPEG75 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 6 |
| Later PNG | 7 / 0 / 1 | 0 / 0 / 6 | 0 / 0 / 0 |
| Later JPEG75 | 0 / 0 / 0 | 0 / 1 / 12 | 0 / 0 / 0 |

Even without a cutoff, the highest-ranked three-layer rectangle was never a useful location. The final-layer rectangles overlapped usefully on 21 of 96 edited deliveries before the cutoff. This separates a ranking failure from an operating-point failure and argues against simply lowering the three-layer point on these inputs. It does not evaluate the trained DINO-FIL network.

A separate replay rebuilt all 384 encoded deliveries and native truth masks and checked 1,152 region measurements and every advancement decision. The replay checks the saved responses and their consequences. Running the fit and confirmation stages reproduces the neural inference.

The public receipt includes the per-view scalar responses and native overlap counts. A standard-library-only command verifies the frozen operating points, sample counts, and advancement decisions using the public file. Exact image and inference verification require the retained sources.

```sh
python3 backend/scripts/review_image_multidepth_public.py docs/evaluation-panels/image-multidepth-transfer-2026-11-11.json
```

## Local replay

Use the retained exact fit intake, Commons source folder, two held intakes, and earlier boundary-blend journal. Pass the pinned local encoder folder and a fresh private output path. No raw photos or neural activations are included in the public receipt.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for stage in freeze fit confirm; do
  backend/.venv/bin/python -B -m scripts.image_multidepth_transfer "$stage" \
    "$checkpoint" "$fit" "$commons" "$earlier" "$later" "$earlier_rows" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_image_multidepth_transfer "$run" "$earlier" "$later"
```
