# Exact edit-region construction with a local image generator

The guarded Mac run completed two 256×256 authored reference edits, one interior and one touching an edge. The raw generator output changed every pixel in both images. After explicit source-coordinate composition, each image had **4,096 changed pixels inside the declared mask and zero outside it**. A separate readback verified all 10 PNGs, reconstructed both change masks from the decoded RGB arrays and confirmed that the outside pixels were byte-identical to the source. [Exact run receipt](evaluation-panels/image-flux-reference-edit-complete-v271.json) · [Frozen inputs and settings](evaluation-panels/image-flux-reference-edit-smoke-freeze-v271.json)

The generator received the source image and a fixed removal prompt. The binary mask was applied to the output in a separate operation. Both authored composites visually replaced the orange rectangle. A later detector comparison should include ordinary edits with the same geometry and composition boundary, since the boundary itself can be visible. This run qualifies input construction on the two authored patterns. It used no real photos and measured no detector accuracy.

The two calls took 48.74 seconds including loading and verification. Sampled process-tree RSS reached 2.68 GiB. MLX reported 5.37 GiB of peak active memory, above the frozen 5 GiB limit for the optional repeat. The two required cases completed and exact repeatability remains unmeasured. [Asset verification](image-flux-klein-assets-v271.md)

The Mac-specific smoke expects the isolated v40 runtime, the exact locally acquired bundle and the existing resource guard. The same frozen code is used in both steps.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.image_flux_reference_edit_v271 freeze
backend/.venv/bin/python -B -m scripts.image_flux_reference_edit_v271 run
```
