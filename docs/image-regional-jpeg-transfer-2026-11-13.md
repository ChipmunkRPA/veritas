# Learned regional editing evidence from one JPEG

The app's aligned luma check measures localized double quantization. A small
generated edit followed by a common JPEG save may leave little such evidence.
The sensitive learned localizer evaluates the decoded image and returns spatial
responses. On the earlier SD2 panel it located 14/59 JPEG75 edits versus 2/59
for the separately tested FUSED localizer, with nine more tone-control
highlights. The two new panels compare the sensitive map with the aligned luma
check on their exact shared bytes. [Earlier model comparison](image-jpeg-diagnostic-v189-result.md)

The native arm keeps its pinned checkpoint, reliability conditions, two-component
evidence cap, and previously fixed 0.90 response and 1% active-area selection
rules. An additional arm tests a second view made by decoding the same upload
and re-encoding it at JPEG90 4:2:0. It keeps each native component's existing
geometry only if the repeat selects at least half of that component's grid
pixels. Both views use the original upload's reliability conditions. No
reference upload, model fitting, new weight download, or authorship
calibration was needed. [Runtime and license record](native-regional-serving-v47-result.md)

## Source coverage

The main panel has 12 photos and 96 JPEG75 4:2:0 deliveries. Six Kandinsky
and six Stable Diffusion 1.5 sources each have sharp and softly blended
versions. The second panel has 64 deliveries from six LaMa removals and one
photograph with separate SDXL and Kandinsky edits. Those two families share a
single second-panel source group. Ordinary controls include untouched images,
blur, local tone and copy, and, where available, local or global autoencoder
processing, resizing and intermediate JPEG encoding. Edit truth uses the
exact pre-final-encode changed-pixel mask. Useful localization requires at
least 0.30 mask intersection over union and 0.50 changed-pixel recovery.

The qualified panels have no shared photo ID, credited Flickr account or
original image SHA-256. None of those identities match any of the five roles
in the local regional adaptation plan. Both panels had earlier project
exposure under other methods. General encoder pretraining membership is
unknown. [Main panel sources and licensing](image-three-class-patch-fresh-2026-11-12.md)
· [Second panel sources and licensing](image-masked-context-new-editors-2026-11-13.md)

## Local result

| Panel | Check | Useful sharp locations | Useful feathered locations | Ordinary file highlights | Ordinary source groups highlighted | Misplaced edit highlights |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Main | Aligned JPEG luma | 0/12 | 0/12 | 0/72 | 0/12 | 0/24 |
| Main | Sensitive regional | 5/12 | 4/12 | 0/72 | 0/12 | 1/24 |
| Second | Aligned JPEG luma | 0/8 | 0/8 | 3/48 | 3/7 | 0/16 |
| Second | Sensitive regional | 1/8 | 0/8 | 5/48 | 2/7 | 2/16 |

The main panel added two useful sharp and one useful feathered location for
Kandinsky and three of each for Stable Diffusion 1.5. The second panel added
one useful sharp SDXL location on the shared diffusion photograph. Neither
check localized any LaMa removal usefully. The two misplaced second-panel
regions occurred on LaMa edit versions. Its five ordinary highlights
occurred across two LaMa source groups and included one original, blur, copy,
tone and resize delivery. The luma highlights occurred on three intermediate
JPEG controls in three groups. The ordinary rates here measure candidate
highlights on known ordinary files.

The recompressed view performed 18 extra forwards and retained every selected
native component. It preserved the useful and misplaced regions and the
ordinary highlights. The optional app check therefore uses the single native
pass.

## Local app scope

Set `VERITAS_IMAGE_REGION_JPEG_REVIEW=true` on the development instance. New Full
and Deep analyses of baseline IJG quality-75 4:2:0 RGB JPEGs with a 768-pixel
long side, a short side of at least 512 pixels and upright orientation can
show the source crop, context, overlay and measured mask. These are editing
cues with zero overall authorship-confidence weight. Other JPEG profiles
retain their existing checks. PNG flip review remains PNG-specific.

An isolated authenticated checkout held the analysis and weight loading
until a simulated $9.99 USD payment cleared. The paid job ran one regional
forward on a held feathered JPEG and reproduced the offline source mask
exactly. The browser displayed the matching source crop and toggled the
measured mask. The evidence API and JSON export returned the same packet and
artifact hashes as the PDF snapshot, and the PDF's Ed25519 signature
verified. The browser, API, JSON and PDF kept the model identity private.
Thirty-five focused regression checks passed. [Evaluation and paid-flow receipt](evaluation-panels/image-regional-jpeg-transfer-2026-11-13.json)

## Reproduce the study

Use the retained main and second-panel assets and the same prior directories as
the [direction comparison](image-directional-structure-transfer-2026-11-13.md#replay).
Use a new run directory. Exact source and truth bindings, native and derived
model-input hashes, native selection masks and all measurements are replayable.
The individual rows and source files remain local.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze run review; do
  backend/.venv/bin/python -B -m scripts.image_regional_jpeg_transfer "$phase" \
    --archive "$archive" --prior-probe "$probe" --residual "$residual" \
    --early-f5 "$early_f5" --later-f5 "$later_f5" --checkpoint "$checkpoint" \
    --output-fresh "$fresh_transfer" --external "$new_editor_transfer" \
    --output "$run" --publication "$receipt" || exit 1
done
```
