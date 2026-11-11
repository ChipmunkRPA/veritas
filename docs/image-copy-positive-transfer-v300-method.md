# Generated and ordinary copy regions on the eight v298 photos

Use the exact eight CC BY 2.0 crops already evaluated on ordinary controls in
v298. Bind its completed public and private receipts and replay the ordinary
view bytes, source coordinates and native RGB change masks before generation.
The source groups were held out for v298 and have already been observed on
ordinary controls. The generated outcomes are prospective on those same
sources. Keep their attribution and inputs in the private evaluation cache.

## Fixed generation and matched mixed edit

Reuse the locally qualified v276/v285 Apache-licensed FLUX.2 Klein 4B revision,
verified 4-bit assets, exact isolated MFlux 0.24.0 runtime, reference prompts,
seed 276004, four steps, guidance 1.0, and 256×256 RGB geometry. The model
receives the reference photo. Compose its raw output afterward inside the
same 64×64 source-coordinate target as the corresponding v298 controls.
Preserve every source pixel outside that target and record the actual
native change mask. A nonempty change verifies construction geometry.
Prompt semantics and visual plausibility are separate questions.

Create a reflected copy at the fixed remote 64×64 box at `(16, 16)` using
the source's existing v298 donor. The pure remote-copy control and the
mixed view use identical copy pixels. The pure generated edit and mixed
view use identical generated pixels. Keep the exact native RGB truth for
both parts separate and verify that the donor remains untouched. Repeat
every submitted view as lossless PNG and whole-image JPEG90 using the
same settings as v298. PNG and JPEG90 share the native truth.

## Stages

```sh
set -e
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run_v300() { backend/.venv/bin/python -B -m scripts.image_copy_positive_transfer_v300 "$@"; }
run_v300 preflight
run_v300 prepare
run_v300 freeze
# Replace the value with the digest printed by freeze.
freeze_sha="printed-digest"
for batch in 1 2 3 4; do run_v300 generate "$batch" "$freeze_sha"; done
run_v300 cue "$freeze_sha"
```

Preflight reads metadata and checks local asset presence and headroom.
Preparation reads only the eight bound sources. Each guarded generation
stage runs two calls offline, under the inherited 6 GiB initial RAM and
sampled worker-tree RSS limits, 3 GiB initial disk reserve, 1.5 GiB running
reserves, and 900-second limit. Schedule these stages serially after any
other neural evaluation completes. The worker checks exact weight bytes.
Use `run_v300 coverage "$freeze_sha"` after a stop. No automatic retries,
substitutions, or replacement based on the generated images are made.

The model-free cue readout binds all four completed generator receipts and
reports 80 planned proposals across both formats. Test the known target
on pure generated and mixed views. Test the remote box on mixed,
remote-copy-only and unchanged views. Keep unique and ambiguous donor
matches, unassessed texture or geometry, exact ordinary donor recovery
and supported generated overlap separate. A generator geometry failure
leaves its positive proposals unassessed while the corresponding
ordinary controls remain measured. The cue preserves every supplied region.

## Current product localization

```sh
for format in native_png jpeg90; do
  for batch in 1 2 3 4; do run_v300 "product-$format" "$batch" "$freeze_sha"; done
done
run_v300 product-seal "$freeze_sha"
```

This optional lane uses the exact v288 production route weights, gates,
reliability, quantization and region selection for PNG and JPEG. Its JPEG
inputs are JPEG90 to match v298. The historical predictor argument
`jpeg75` selects the JPEG route only. It does not encode these inputs.
The current PNG CPU and JPEG MPS routes retain their existing operating
points. This is a readout of the current product routes, independent of
the optional sensitive candidate.

Each of eight guarded stages processes two source groups and one format.
All eight variants are retained per source and format, for 128 planned
views and two exact first-view repeats when all inputs qualify. The
separate native masks support independent measurements of generated
localization and ordinary overlap. Report useful generated localization
at the prior IoU and recall requirements for both the complete selected mask
and its individual components. Report product views with no highlight and
copy corroboration on each retained source-mask component. Confirm
that the full product mask remains byte-identical after cue evaluation.
Never use a donor to remove an AI evidence region or issue an authenticity
or authorship verdict.
