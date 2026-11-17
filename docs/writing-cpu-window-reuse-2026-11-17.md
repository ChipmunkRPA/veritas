# Local CPU analysis for longer documents

A public seven-page Federal Register PDF exhausted the 180-second
local analysis budget. Its passage classifier finished, and the paired
writing-pattern check exhausted the remaining 119 seconds. The model
manager had a 7,600 MiB allowance, so each rolling window of up to
1,024 tokens loaded the two writing models in sequence.
[Federal Register source](https://www.govinfo.gov/content/pkg/FR-2025-06-12/pdf/2025-10477.pdf) ·
[Pinned source commitments](evaluation-panels/pdf-saved-wording-sources-2026-11-17.json)

The local CPU fallback now holds one model for several windows. It
retains each window's FP32 hidden states and an exact representation
of each output head. The worker confirms each head's full FP32 bit
pattern after widening the original BF16 values. The model manager
reserves this memory before the forwards. After the two backbone
leases finish, the worker uses the original FP32 projections and
loss reductions. The prior sequential path can serve single windows,
incompatible weights, and cases where the grouped reservations are
unavailable. The two pinned checkpoints are listed as Apache-2.0 in
the model manifest.
[Paired writing method](https://proceedings.mlr.press/v235/hans24a.html) ·
[Pinned model manifest](../backend/app/models.yaml)

## Numerical replay

Both paths ran with FP32 CPU computation, two threads, and the same
7,600 MiB model allowance on the Mac. The longer calculation budget
let the earlier route finish so that every token loss could be compared.
The web worker runs below used the original 180-second analysis budget.

| Input | Scored tokens | Windows | Model loads, earlier → grouped | Check time, earlier → grouped |
| --- | ---: | ---: | ---: | ---: |
| Federal Register PDF | 11,843 | 16 | 32 → 2 | 215.72 s → 94.47 s |
| Ordinary news and report excerpts | 832 | 2 | 4 → 2 | 24.99 s → 15.79 s |
| Matched excerpts with AI rewrites | 839 | 2 | 4 → 2 | 23.07 s → 16.99 s |

Each pair has identical rolling inputs, observer loss arrays,
cross-entropy arrays, document response, and scored sentence records.
The edited file includes publisher-labeled rewrites from two generators.
It joins one pair from each genre and generator combination in sorted
genre and generator order. The selection uses the greatest group key
from each combination in the pinned prior test plan. The exact text
comes from that plan and the public receipt contains its commitments.
[Text plan and publisher terms](writing-structured-range-transfer-2026-11-15.md#sources-and-coverage)

## Prepaid app and source replay

The previously timed-out PDF completed in 159.893 seconds. Its final
writing evidence matches the numerical replay across all 336 scored
sentences. A second PDF, the ordinary text, and the AI-edited counterpart
also exercised the prepaid worker. The four cases delivered the
writing check's coverage to JSON and PDF. The signed exports were
verified with their submitted originals. The checkout used simulated
USD 9.99 payments and recorded zero jobs before payment.
[Second Federal Register source](https://www.govinfo.gov/content/pkg/FR-2025-06-12/pdf/2025-10609.pdf) ·
[Validation receipt](evaluation-panels/writing-cpu-window-reuse-2026-11-17.json)

This replay measures runtime and numerical parity for the documented
files on the tested configuration. The optional
`VERITAS_TEXT_RATIO_COMPACT_CPU=false` setting restores the earlier
local CPU scheduling.

## Reproduce

Set `source` to the first Federal Register PDF above and use the
repository's locally installed verified weights. The standard-library
reader verifies the recorded pair and app commitments.

```sh
export PYTHONPATH=backend
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=2
run=$(mktemp -d)
panel=docs/evaluation-panels/writing-cpu-window-reuse-2026-11-17.json
runner=backend/.venv/bin/python

python3 -B -S -m scripts.replay_long_document_cpu "$panel" --verify-panel
"$runner" -B -m scripts.replay_long_document_cpu "$source" "$run/earlier" --per-window
"$runner" -B -m scripts.replay_long_document_cpu "$source" "$run/grouped"
python3 -B -S -m scripts.replay_long_document_cpu "$run/earlier/receipt.json" "$run/grouped/receipt.json" --verify
"$runner" -B -m scripts.replay_long_document_cpu "$source" "$run/app" --app
```
