# Generated replacements in native camera renders

This pilot isolates genuinely synthesized content inside a known local
footprint. Six public-domain camera RAW captures from distinct camera
models supply the underlying photographs. Those cameras were exposed
earlier under conventional copy and interpolation tests. Their generated
edits, source placements and mask composition are new. The local
generator is the pinned Apache-2.0 FLUX.2 Klein 4B bundle. The experiment
retains each verified generator output and uses no external inference API.
[CC0 source roster](evaluation-panels/image-native-cfa-cc0-png-roster-2026-11-13.json) ·
[Generator weights and license](image-flux-klein-assets-v271.md)

The published DiffusionPrint work motivated separating a synthesis cue
from general source-residual anomalies. Its publisher supplies a
CC BY 4.0-tagged full TruFor checkpoint, while upstream TruFor code is
limited to nonprofit purposes. This pilot uses its own analysis code and
uses no DiffusionPrint or TruFor software or checkpoints.
[DiffusionPrint method](https://arxiv.org/abs/2604.12443) ·
[Publisher checkpoint](https://huggingface.co/giakoupg/diffusionprint) ·
[TruFor terms](https://github.com/grip-unina/TruFor#license)

## Frozen design

Develop each RAW to its existing native-grid 16-bit linear RGB crop.
The generator sees a 256 × 256 display-space reference with a fixed
instruction to insert a matte red paper disk. Copy its central 128 × 128
area back into the photograph using a fixed code-value transfer to
16-bit linear. The hard and eight-pixel feathered versions use the same
generated result. Original pixels outside the known target remain
identical. Ordinary controls use local tone, blur, even-aligned copy,
fractional interpolation and classical fill at the same location.

Run the existing phase-shift score, an experimental phase-loss score,
and phase loss limited to cells whose local high-frequency activity is
at least half their neighborhood level. The loss comparison uses the
same prior operating point in both loss arms. The texture requirement
tests whether ordinary blur can be rejected without also suppressing
generated replacements. A useful box must cover at least half the exact
changed pixels and reach 0.30 intersection over union. The measured
region shows an interpolation anomaly and carries zero authorship weight.
[Earlier phase calibration](image-native-cfa-closure-2026-11-13.md)

## Held result

| Check | Useful generated sharp | Useful generated feathered | Ordinary local edits highlighted | Untouched files highlighted |
| --- | ---: | ---: | ---: | ---: |
| Existing phase shift | 0/6 | 0/6 | 0/30 | 0/6 |
| Phase loss | 6/6 | 6/6 | 11/30 | 0/6 |
| Phase loss with texture support | 5/6 | 5/6 | 6/30 | 0/6 |

The generator placed the requested red disk visibly within the masked
area on all six source images in a visual check. Each composite changed
exactly 16,384 pixels and retained every pixel outside the mask. Every
selected generated region matched the changed-pixel footprint exactly.
Phase loss highlighted five blur and six fractional-interpolation
controls. Texture support reduced those to one and five, respectively.
It suppressed the sharp and feathered result on one source. Neither
loss arm reached the frozen maximum of one ordinary local highlight.
Phase loss is a useful location cue on this pilot, with fractional
interpolation a major ambiguity. It remains experimental. The panel
covers one generator, one conspicuous edit instruction and a narrow
native-development workflow. Native replay verified all 48 input and
mask commitments, scores and boxes.
[Audited results](evaluation-panels/image-native-generated-stencil-transfer-2026-11-13.json)

## Replay

Use the prior verified RawNIND decoder cache, the CC0 source cache,
the local model bundle and the archived MLX runtime. The data score
and review are CPU-only. The generator stage needs local Metal access. The restored runtime
needs package files free of macOS AppleDouble metadata sidecars.
Use a fresh run directory for the freeze. Generation binds output bytes
to the chosen seed and source crop. A replay reconstructs every exact
mask and response from the retained output bytes. An independent
generator repeat is a separate test.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
backend/.venv/bin/python -B -m scripts.image_native_generated_stencil_transfer freeze \
  "$run" --cached "$rawnind" --archive "$pixls"
"$cpython3126" -B -m scripts.image_native_generated_stencil_transfer generate "$run" --runtime "$mlx_runtime"
for phase in score review; do
  backend/.venv/bin/python -B -m scripts.image_native_generated_stencil_transfer "$phase" \
    "$run" --cached "$rawnind" --archive "$pixls" --receipt "$receipt" || exit 1
done
```
