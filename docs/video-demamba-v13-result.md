# General-video temporal development result

7 November 2026. **Rejected; current serving checks retained.** The complete
consumed-source study contains288 planned variants:96 calibration and192
development variants, including unavailable and unassessed inputs. It does not
establish fresh-source, face-swap, lip-sync or whole-file service performance.

| Development outcome | Current synthesis adapters | Candidate with own calibration |
| --- | ---: | ---: |
| AI correctly flagged, of96 AI variants | 51 | 0 |
| Original correctly identified, of96 originals | 25 | 0 |
| Original falsely flagged as AI | 5 | 0 |
| AI falsely called original | 8 | 0 |
| Inconclusive, of192 planned variants | 103 (53.6%) | 192 (100%) |
| AI without an AI conclusion, including inconclusive | 45 | 96 |
| Assessed inputs | 175 | 188 |

Zero wrong-direction calls from an always-inconclusive method is not a useful
accuracy gain. Six unchanged pre-native requirements failed: Brier improvement,
its paired interval, useful AI calls, original-call retention, total correct
calls and retention of baseline assessment coverage. Distinct AI originals
correctly flagged fell from37 to0. The final native requirement failed too.
No fresh confirmation clips were consumed to rescue this result.

## Calibration and continuous scores

Both systems received their own nonnegative-slope Platt calibration on the same
assessed calibration inputs, separately by source collection. DeepAction's
candidate slope reached0; GenBuster's was0.04598. These mappings yielded no
directional candidate decisions under the frozen research operating points.
Those points are research comparisons, not a public accuracy or confidence
promise. Development labels were not used to reverse polarity or change points.

| Panel | Candidate minus current Brier loss | Source-group bootstrap95% interval |
| --- | ---: | --- |
| DeepAction native | +0.20053 | [0.15612, 0.23303] |
| DeepAction matched | +0.07842 | [-0.01727, 0.15339] |
| GenBuster native | +0.14485 | [0.10953, 0.17885] |
| GenBuster matched | -0.05505 | [-0.13285, 0.02566] |

Lower loss is better; the one improved panel's interval still includes zero.
Post-result inspection found no clipped candidate logits and confirmed the
publisher's real0/fake1 label convention. Poor transfer or weak calibration
separation is plausible, but its cause has not been isolated. We did not retune
the method after inspecting these outcomes. GenBuster parent/generator separation
is not established; neither collection is claimed training-disjoint or fresh.

## Resource completion and native correspondence

The initial84-row OS termination and later42-row preset memory-guard stop remain
separate, incomplete attempts. A six-input diagnosis motivated explicit cyclic
object collection after each unchanged scorer call. Its operational freeze was
published as `ac5f6d1d7b24dcab382ee312dc9041c838b5a806` before the new full run.

That new run completed all288 baseline rows (264 assessed,18 unassessed,
four unavailable), all284 ready-input cleanup records and all84 prior input
comparisons with exactly equal raw responses and assessment statuses. No resource
guard triggered; the child exited0 and its completed receipt was checked.
There was no model/input/scoring/calibration or comparison-rule change.

The tree sampler observed4,594,237,440 bytes; its record was incomplete because
some child observations failed. This is a sampled lower bound, not a measured
peak or proof of maximum memory use. The original termination cause remains
unisolated. The operational fix is in the research wrapper, not the serving app.

The unchanged16-case native audit completed under the same memory limits after
its own public freeze, commit `637e3a35c162535ceb5c3890eaae9fd984e6c2aa`.
Twelve cases passed. Four cases (DeepAction010/074 Pexels originals and their
matched versions) failed exact RGB correspondence between OpenCV and separate
FFmpeg capture. Their cause is unresolved. Those cases did not complete their
later neural-response replay. No tolerance was widened and no failed case was
substituted. The native sampler was also incomplete. A successful arithmetic
review validates reported calculations and failure handling; it cannot turn
these native failures into passes.

## Public-only reproduction

The [lossless packet](evaluation-panels/video-v13-development-review.json.gz)
contains nine exact UTF-8 members: complete candidate/baseline journals, summary,
284 collection events, both parent/child execution receipts and all16 native
outcomes. It contains no raw videos, model weights, customer uploads or private
machine paths. Original member byte lengths and SHA-256 checksums are retained.
Packet SHA-256:
`694d537982f98b25c7695e2ec66e0a19e4a0935691c8e76a407e544502969472`.

```sh
python -m pip install -r backend/requirements-video-public-review.txt
PYTHONPATH=backend python -m scripts.review_video_public_v13 review \
  docs/evaluation-panels/video-v13-development-review.json.gz \
  /tmp/video-review.json \
  --freeze docs/evaluation-panels/video-v13-demamba-freeze.json
```

This separate NumPy reducer reproduced all192 outcomes per method, rank metrics,
losses, grouped intervals and all11 final requirements in an isolated numerical
environment. It uses stored calibration coefficients, not an independent refit;
it binds the native receipts without rerunning video decoding or neural models.
The original prospective freeze checksum is embedded in the arithmetic reviewer.
[Completed local receipt](evaluation-panels/video-v13-independent-review.json).
Fourteen existing arithmetic integrity tests passed. The public workflow adds
an external Linux replay; its result must be reported separately when available.
Actual outside human reviewers remain zero. Current detector, overall confidence,
customer reports, prepaid billing and GCP infrastructure are unchanged.
