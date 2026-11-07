# Video synthesis development execution

The complete [protocol](video-demamba-v13-protocol.md) and
[prospective freeze](evaluation-panels/video-v13-demamba-freeze.json) bind the
comparison before any project recording is scored. This is a consumed-source
development study, not fresh confirmation or a production model replacement.

The lock retains all 288 planned inputs: 284 available and four already
unavailable, with 96 calibration and 192 development variants. It pins 360
project implementations, eleven critical installed files/binaries, complete
candidate and baseline weights, original source manifests, roles, preprocessing,
calibration, comparison requirements and sixteen response-independent native
replays. No raw video, model weights or customer uploads are published.

Before recording inference, 53 selected tests passed. They include a separate
FFmpeg capture of a generated lossless clip that reproduces OpenCV's native
pixels, frame indices, timestamps and normalized inputs. The complete real-data
lock was then checked unchanged, and eleven altered copies were refused:
threshold, native selection, project code, runtime, installed code, source
truth, preparation receipt, prior scoring, prior fitting, serving admission
and model-asset commitments. The [review receipt](evaluation-panels/video-v13-demamba-freeze-review.json)
records these checks. These safeguards establish reproducibility boundaries;
they do not measure detection quality.

The public freeze must be pushed and its remote commit verified before starting
either recording journal. The candidate runs on CPU; the current synthesis
adapters run on MPS, their existing local acceleration. Their runtimes cannot
establish a speed comparison across hardware. Any unsupported input remains
unassessed, and any unexpected failure stops its journal and is retained.

No serving defaults, overall confidence, customer evidence, payment settings or
reserved confirmation sources change at this stage. Independent arithmetic
review and actual outside human testing remain separate requirements. GCP
deployment remains last.

## Separate arithmetic review

`backend/scripts/review_video_demamba_v13.py` is an additive, NumPy-only review
path. It imports neither the app nor the original evaluator. Using the recorded
calibration coefficients, it separately reconstructs all 192 development
outcomes per method, common calibration support, tied rank metrics, losses,
source-group bootstrap intervals and every comparison requirement. It binds the
sixteen native replay receipts without rerunning their media or neural models.
It does not independently refit calibration or establish the truth of source
labels. NumPy's random generator and quantile implementation remain shared.

Fourteen tests passed, including a complete synthetic 288-input comparison,
hand-calculated tied rank metrics, retained missing cases, and eleven altered
packet checks. This is preparation for public arithmetic reproduction; it is
not a detection-quality result or participation by an outside human reviewer.

After the recording journals, comparison and native receipt are complete:

```sh
python backend/scripts/review_video_demamba_v13.py \
  docs/evaluation-panels/video-v13-demamba-freeze.json \
  baseline.jsonl candidate.jsonl summary.json native-review.json arithmetic.json
```

The public freeze's exact checksum is embedded in the reviewer. Arithmetic
comparison allows relative error1e-11 and absolute error1e-12; directional
outcomes and requirement decisions must match exactly. This tolerance does not
alter the original model, calibration, native replay or advancement rules.
