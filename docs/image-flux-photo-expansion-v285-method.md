# Licensed photo edit expansion

Select six additional CC BY 2.0 source groups from the retained v39 JPEG
cache. Reuse the v276 attribution, exact-byte, and overlap audit and exclude
both complete groups already used by the v276 construction and v280 detector
comparison. The next six groups follow the existing salted file-digest order.
The audit covers the retained still-image exposure records. It cannot establish
absence of creator aliases, transformed duplicates, unrecorded exposures, or
model pretraining overlap.

Take one native 256×256 center crop per source. Three paired batches each have
one 64×64 interior selection and one left-edge selection. Reuse the v276 prompts,
seed 276004, four steps, reference-edit model and offline runtime. The raw edit
receives the reference image. The worker applies the mask afterward and verifies
that the composed image preserves every pixel outside the declared selection.
Tone, blur, and mirrored copy controls share the same source and boundary.

```sh
set -e
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run_v285() { backend/.venv/bin/python -B -m scripts.image_flux_photo_expansion_v285 "$@"; }
run_v285 preflight
run_v285 prepare
run_v285 freeze
for batch in 1 2 3; do
  run_v285 run "$batch"
done
run_v285 seal
```

The freeze binds all six prepared inputs before any model call. Every two-call
batch inherits the v108 resource limits and isolated v40 runtime checks. Use
`run_v285 coverage` after an interruption to inspect verified, unverified, and
unattempted batches. The seal requires all six outcomes and independently
reconstructs each successful composition and actual mask. Prepared sources,
credits, controls, and generated files stay in the private evaluation cache.
The aggregate qualifies construction geometry. It does not measure semantic edit
success, detector accuracy, or confidence calibration.
