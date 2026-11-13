# Sustained writing responses across consecutive sentences

The full report benchmark shows that short classifier inputs rarely cover
enough of a larger revision. A reference-bound diagnostic found that at
most 23 of 105 edited reports could satisfy the strict overlap rule with an
existing three-sentence window. A freely positioned contiguous range
could meet it on 100 of 105. These figures describe the benchmark's geometry.
They use known edit locations and are not detector results.
[Complete worker benchmark](writing-revision-worker-transfer-2026-11-13.md)

The offline comparison tests a local change-point idea from the published
research using the app's recorded inputs and responses. It scans runs of
three to eight consecutive centers in one paragraph, with two measured
centers on each side. A run must stay elevated over both flanks at two-thirds
of its positions. The score is the difference between the interior median
and the stronger flank. One arm uses classifier log odds. The other requires
the local writing ratio to support the same run and uses the weaker contrast.
The two readings use separate robust scales and the ratio arm leaves out
expanded or missing contexts.
[Local-evidence change-point paper](https://arxiv.org/abs/2609.33787)

The candidate point for each arm was locked from the 16 genuine fit files
before running the new selector on the held ledger. The held text, underlying
scores, and prior app outcomes had project exposure. This trial tests the
new selection rule on the retained readings without new inference. A strict
location covers at least half of the labeled non-whitespace characters and
reaches at least 0.30 intersection over union.

| Result | Existing app | Sustained classifier | Sustained agreement |
| --- | ---: | ---: | ---: |
| Changed versions assessed | 105/105 | 105/105 | 99/105 |
| Strict locations | 4/105 | 1/105 | 0/105 |
| Strict locations missed by the app | — | 1/105 | 0/105 |
| Half of any one annotated region covered | 92/105 | 21/105 | 9/105 |
| Genuine originals assessed | 20/20 | 20/20 | 19/20 |
| Genuine originals highlighted | 20/20 | 5/20 | 1/20 |

The single strict range came from one generator family. Neither arm
reached the predefined three-source, two-family gain with at most two
genuine highlights. The candidate checks remain offline. The two-reading
arm's reduced review volume also came with reduced usable coverage and
zero strict locations.
[Recounted scalar results](evaluation-panels/writing-sustained-response-2026-11-13.json)

The reader recomputes both candidate points and all held locations from
the published worker ledgers using the Python standard library. Each
retained positive range maps to exact extracted text offsets. The source
prose stays outside the receipt.

```sh
scratch=$(mktemp -d)
export PYTHONPATH=backend
python3 -B -S -m scripts.writing_sustained_response_transfer freeze \
  docs/evaluation-panels "$scratch/freeze.json"
python3 -B -S -m scripts.writing_sustained_response_transfer review \
  docs/evaluation-panels "$scratch/freeze.json" \
  docs/evaluation-panels/writing-sustained-response-2026-11-13.json
```
