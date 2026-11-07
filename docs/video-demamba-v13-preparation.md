# Temporal video candidate: strict loading and synthetic checks

The separately published
[DeMamba/GenVideo-100K checkpoint](https://huggingface.co/Ahsanz/DeMamba-GenVideo100K/blob/08664b1ea90a1de2bbc1f9a77e3ba0dd1dd4846e/README.md)
has been acquired for a local comparison of general AI-generated video. It has
not scored project recordings, been calibrated, passed predictive evaluation
or entered the service. This preparation does not establish face-swap or lip-sync
detection, timestamp localization, fresh accuracy or outside human review.

## Assets and permissions

The publisher declares its own fine-tuned weights under Apache-2.0. Its
[X-CLIP backbone card](https://huggingface.co/microsoft/xclip-base-patch16/blob/d6184e3fd8780d04c85d0f1eabe5f94bf44d98f6/README.md)
declares MIT. Retain the separate upstream
[DeMamba Apache-2.0 notice](https://github.com/chenhaoxing/DeMamba/blob/646db285f4b473b2678d9a9eeb836f781459fa99/LICENSE),
[Mamba implementation MIT notice](https://github.com/alxndrTL/mamba.py/blob/5c4136c2c28bc10d5d6655aa81b7a60051b50f00/LICENSE)
and [VideoX notice](https://github.com/microsoft/VideoX/blob/1744b2159c3d11694f94dab4fcde8660c685e79a/LICENSE).
The source/weight declarations permit a commercial-use-capable research path;
they are not an independent audit of every training item or underlying content
right. Source/pretraining overlap with future evaluation inputs remains unknown.

The full 502,986,967-byte checkpoint matches publisher SHA-256
`f89897c72b595aa85a524e59083919cebb0b92161d2a694a2cbffbdd8456ce17`.
It includes the fine-tuned vision encoder; no separate pretrained backbone was
downloaded. No packages were installed into the app. Source, configuration,
declarations and notices are pinned beside the local research asset; full
weights are not committed to GitHub.

The [previous joint audio-visual candidate](audio-visual-v12-source-review.md)
was excluded after its required backbone declared non-commercial terms. A
different EfficientNet release was also deferred when its weight card declared
CC-BY-NC-4.0 despite MIT code. The
[permission record](evaluation-panels/video-v13-source-permissions.json)
retains these distinctions. No restricted weights were acquired for this turn.

## Strict offline loader

The research loader verifies every source/notice and the complete checkpoint
before construction. It executes nine inspected model/scan definitions from the
publisher's pinned standalone file, excluding its imports, downloader, data
loader, decoder, metrics and command entry point. It scans checkpoint globals
and deserializes with `weights_only=True`; the unsafe-globals list is empty.

All 366 tensors, including 125,710,849 parameters, load strictly. A bijective
DataParallel/vision-prefix mapping preserves tensors and refuses collisions,
unknown keys or missing state. CPU float32 and eager attention are explicit.
The publisher's channel-axis reversal is preserved; changing it would define
a different model. Full training-runtime equivalence has not been demonstrated.

The initial local loader left a nonpersistent position-index buffer on the
meta device. The first synthetic attempt returned non-finite outputs and failed;
its code/log and diagnostic were retained locally. First-loader SHA-256:
`b04eb01ca3ae93a969b0d5d064e6f5e8de9309127e7f95999b5de4917c165207`.
The fix reconstructs only the inspected constructor constant `arange(197)` on
CPU. Every trained tensor still comes from strict checkpoint assignment. Other
uninitialized buffers or corrupted indices are refused. Eight focused tests
cover these failure paths, bijective mapping and hand-computed recurrence.

## Completed synthetic verification

The [complete receipt](evaluation-panels/video-demamba-v13-synthetic-review.json)
records all hashes, runtime, mappings, outputs and original tolerances. Nine
small scan probes, lengths 1–31, agree with an independently written serial
recurrence within 0.000000477. Three complete synthetic graph inputs—zeros,
seeded noise and a temporal ramp—produce finite repeat-exact outputs. Replacing
only the scan with that direct recurrence changes the largest final logit by
0.000000954, within the originally declared absolute 0.0001/relative 0.00001
graph tolerances. No tolerances were relaxed after the failed initialization.

The independent reference covers recurrence arithmetic. The vision library,
trained tensors, remaining graph and framework remain shared. Synthetic outputs
are neither calibration examples nor authentic/AI test cases. Their high raw
logits must not be represented as detection accuracy or per-upload confidence.

From `backend`, using the locally acquired pinned directory:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m scripts.video_demamba_v13 \
  /path/to/video-v13-demamba /tmp/new-video-v13-synthetic-review.json
```

## Required comparison before any adoption

Prepare and publish a complete prospective execution freeze before project
video forwards. Use consumed development inputs first, preserving native and
matched-encoding counterparts, source groups, unavailable files and all ordinary
controls. Bind the current synthesis checks, candidate runtime, every source and
all preprocessing/calibration/decision code before scoring. Do not treat
GenVideo, on which this candidate was selected, as fresh confirmation.

The planned comparison must retain actual decoded frame times and pixels,
declare the inspected sampling window, distinguish spatial evidence from time,
and abstain on missing/invalid support. Missing checks cannot become original
verdicts. The publisher's default score is uncalibrated; fit only declared
calibration sources, report paired probability errors, false calls, missed AI,
useful conclusions and inconclusive rates, and preserve every frozen condition
and adoption requirement. No post-result threshold search or selective routing.

Only a passing consumed comparison permits separately frozen, source-disjoint
fresh confirmation. Customer evidence/PDF/JSON/prepaid-flow checks and separate
overall-confidence validation would follow. The running audio separation trial,
image/writing improvement and actual outside testing remain active. GCP stays last.
