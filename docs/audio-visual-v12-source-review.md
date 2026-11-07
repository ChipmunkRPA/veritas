# Audio-visual candidate: resolve loading and time axes before evaluation

The publisher supplies a commercially usable MIT declaration for
[DAVID-Net's code](https://github.com/MIHMahmudEli/david-net-av-v2/tree/bc911636a11a42fd0291bc31a5f092e1dd3dc811)
and [weight card](https://huggingface.co/MIHMahmudEli/david-net-av-v2/blob/a071972e6b9325da99a0d4ec2adce35a2446215b/README.md).
It is worth inspecting because it separates audio, picture and joint outputs.
**Do not adopt its supplied API or count this as a detection improvement.**
The local preflight found source-level loading concerns and reproduced a
spatial/temporal index problem in its reporting conversion. No customer code,
confidence, calibration, payment or infrastructure changed.

## What was checked

The [receipt](evaluation-panels/av-v12-source-preflight.json) binds the exact
publisher revisions and seven small source/declaration files. Only the
17,480-byte SafeTensors prefix/header was acquired from the checkpoint, not its
341,484,140-byte tensor payload. The declared 187 tensor names, shapes, float32
types and contiguous offsets agree with an inspected head graph instantiated
on the meta device: 85,366,665 parameters from shapes. Identity encoders supply
no backbone parameters in this structural check.

This establishes header/architecture compatibility. It does not verify the
complete checkpoint checksum, load trained tensors, run a numerical model
forward or demonstrate that the weights learned useful behavior. VideoMAE and
WavLM backbones were not acquired or loaded for this check. No recording was
scored and no fresh subject was consumed.

Only checksum-verified model definitions and the isolated interval helper were
executed. The full publisher package, API initialization, checkpoint loader,
decoder and remote scripts were not executed. Eight focused tests passed,
covering source/header alteration refusal, bounded headers, selected-definition
execution, immutable acquisition URLs and refusal to overwrite a prior receipt.
These are implementation checks, not accuracy samples or outside human review.

## Reproduced timestamp problem

The [pooled encoder](https://github.com/MIHMahmudEli/david-net-av-v2/blob/bc911636a11a42fd0291bc31a5f092e1dd3dc811/src/pipeline/encoders.py)
flattens eight temporal groups and sixteen spatial patches per group into 128
tokens. The [supplied interval helper](https://github.com/MIHMahmudEli/david-net-av-v2/blob/bc911636a11a42fd0291bc31a5f092e1dd3dc811/api/inference.py)
then assigns every token its own uniformly spaced time interval. In an isolated
four-second synthetic example, changing which patch has a high logit produces:

| Flat token | Temporal group | Spatial patch | Nominal group bin | Reported interval |
|---|---:|---:|---|---|
| 0 | 0 | 0 | 0.00–0.50 s | 0.00–0.03 s |
| 15 | 0 | 15 | 0.00–0.50 s | 0.47–0.50 s |
| 16 | 1 | 0 | 0.50–1.00 s | 0.50–0.53 s |
| 31 | 1 | 15 | 0.50–1.00 s | 0.97–1.00 s |

The interval helper's actual outputs match these calculations. Spatial patch
position therefore changes a reported boundary inside one temporal group. There
are eight temporal groups, not 128 separate acquisition times. These nominal
bins are not independently verified source clocks or causal edit boundaries;
global transformer context also extends beyond a token's nominal bin. The test
does not measure trained localization accuracy. It identifies a reason to
preserve spatial and temporal axes separately before producing user evidence.

## Loading and provenance concerns

Source inspection of the same pinned API shows that failure to obtain a
checkpoint can leave an initialized model available for prediction; its
lightweight path also constructs an initialized head. If foundation backbones
fail to load, another path substitutes pooled RGB/audio features instead of the
learned representations. These paths were inspected, not executed here. Neither
is an acceptable substitute for the validated detector in Chipmunk Verify.
Any candidate adapter must abort on missing weights, incompatible tensors or
missing backbones and retain an unassessed outcome.

The supplied configuration and historical revision record point to other
repositories/revisions. Equivalence between that historical training code and
the inspected public checkout has not been verified. The MIT declarations do
not independently resolve every training-data or backbone right. No training
corpus was downloaded, and publisher benchmark numbers were not adopted as
our accuracy or confidence calibration.

## Reproduce and continue

From the project root, with the existing local research environment:

```sh
backend/.venv/bin/python backend/scripts/review_av_candidate_v12.py acquire \
  /tmp/new-av-v12-source-inputs
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  backend/.venv/bin/python backend/scripts/review_av_candidate_v12.py review \
  /tmp/new-av-v12-source-inputs /tmp/new-av-v12-preflight.json
```

Acquisition requests only pinned public source files and the bounded tensor
header. Review uses those local files and PyTorch; it performs no network
requests or source-media inference. Existing sources are verified before reuse,
and existing reports are never overwritten.

Next establish complete weight/backbone lineage, implement a strict research
adapter with decoded picture/audio clocks, and freeze the complete comparison
before any source scoring. Retain all four picture/audio authenticity
combinations, ordinary edits, false flags, missed locations and inconclusive
results. A passing development result would still need fresh confirmation,
own calibration and evidence/export/prepaid-flow verification. The current audio
separation experiment continues separately. Outside human reviewers remain zero;
image, writing, video improvement and eventual GCP validation remain unfinished.
