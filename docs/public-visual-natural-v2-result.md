# Natural-media watermark recovery and its remaining limits

The [prospectively frozen protocol](public-visual-natural-v2-protocol.md)
completed **640 cases / 709 neural forwards** on 32 source photographs and eight
camera clips. Every deliberately marked photo/frame was flagged at the unchanged
inspection point. However, no-added-mark controls also flagged, compressed
partial-message recovery degraded, and seven compressed frames flagged outside
the intended marked interval. **This public-family check remains outside serving,
AI confidence and the early-stop gate.**

These are known added-watermark experiments on source-reviewed natural media,
not a test of whether arbitrary scenes were AI-generated. Public marks can be
added to genuine material. Their presence does not authenticate a provider,
creator or the origin of an entire file. The existing trusted, current-file-bound
Content Credentials gate is unchanged.

## Frozen scope and source evidence

Code and the prospective freeze were published and remotely verified at
`257e04aecfc70e980f474fbd05bd3d951ab673e7` before model scoring. The
[public freeze](evaluation-panels/public-visual-natural-v2-freeze.json) commits
to the private freeze SHA-256
`b48e71f0b97bfe1fff07d2f183159058187bac3cc3951f555159467f39bc6a2f`.
All 14 protected code/test/protocol files, source-plan/acquisition/visual-review
checksums, package versions and model/source/dependency identities were fixed.
No source replacement, dropped case, point change or repeated model run occurred.

The panel contains ten architecture, ten food, six plant and six landscape
photos from 25 credited creators, plus eight clips from eight credited creators.
Per-file rights metadata records 27 CC BY-SA 4.0, 11 CC BY-SA 3.0 and two CC BY 3.0
sources. Individual attribution and change records remain attached privately.
Source dates/licenses are publisher claims; the cohort is fresh to the known
project URL inventories, with pretraining overlap and independent capture truth
unestablished. No raw sources, case pairings or prediction arrays are published.

The exact downloaded thumbnails/transcodes, center-crop geometry and original
video PTS are retained. Only the 256-square working inputs and eight sampled
frames per clip were scored. Full-resolution originals, unsampled video frames
and audio were not examined. All 40 sources and 56 visual previews were reviewed
for photographic eligibility before inference. Four fixed 32-bit payloads test
message diversity; their choice was not fitted to outcomes.

## Images

Each denominator is the same 32 source photos under the named condition.
Variants share sources; they are not independent population samples.

| Condition | No-added-mark flags | Ordinary-edit flags | Full marks missed | Partial marks missed | Partial mean IoU | Exact partial message |
|---|---:|---:|---:|---:|---:|---:|
| Native | 2/32 | 1/32 | 0/32 | 0/32 | 0.9856 | 32/32 |
| JPEG quality 70 | 3/32 | 2/32 | 0/32 | 0/32 | 0.8483 | 3/32 |

All native full marks had exact message recovery and mean IoU 1.0000. After
JPEG, full-mark mean IoU was 0.9916 and 29/32 full messages recovered exactly.
For partial JPEG marks, the mean recovered-bit accuracy among available messages
was 0.9141, while exact 32-bit recovery was only 3/32. Average bit accuracy must
not be mistaken for reliable payload attribution. One partial JPEG case failed
the predeclared flagged-region IoU≥0.5 localization summary, despite flagging.

Native no-added-mark flags are 6.25% of this 32-photo condition; JPEG flags are
9.375%. These are observed control counts, not calibrated population error rates.
The control images did not receive this study's watermark; unknown upstream
processing or marks are not independently certified absent. A high-recall public
family inspection rule can still mislead if it is presented as AI authorship.

## Video frames and sampled time intervals

Each condition contains eight clips / 64 sampled frames. The partially marked
condition has 32 deliberately marked frames and 32 frames outside that interval.

| Condition | No-added-mark flags | Ordinary-edit flags | Marked frames missed | Outside-interval flags | Marked mean IoU | Exact marked-frame message |
|---|---:|---:|---:|---:|---:|---:|
| Raw sampled sequence | 1/64 | 1/64 | 0/32 | 0/32 | 0.9890 | 32/32 |
| H264 CRF 23 | 0/64 | 0/64 | 0/32 | 7/32 | 0.8661 | 15/32 |

The raw no-added-mark and ordinary-edit flags each involve one of eight clips.
All eight marked clips had flagged frames inside the intended interval under
both codecs; no clip's entire marked interval was missed. H264 outside-interval
flags occurred in **four of eight clips**, including three at the BEFORE sample,
one at the first AFTER sample and three at the last AFTER sample. Reporting only
“a marked clip was detected” would hide those timing errors.

The original source sample times and the derived 2fps presentation times are
different clocks. For example, a derived 0.50s preview can map to an original
source timestamp of 0.507s. Both are preserved in the private evidence journal.
Before/during/after overlays and matched source/inspection-region crops were
rendered and visually reviewed locally, including false image flags, the weakest
partial-photo localization and the clip with the most outside-interval flags.

Partial insertion changes no pre-codec pixels outside the known interval/region.
After inter-frame compression, a family response outside that interval can be
propagation of watermark-related codec changes or a false-family response.
This study does not isolate that cause. It cannot turn those timestamps into
proof of an AI edit. Exact H264 message recovery also fell to 15/32 marked frames,
so recovered payloads cannot be assumed reliable merely because a region flags.

## Replay, resources and publication

The separate model-free audit independently re-decoded all 40 source assets and
64 original sampled-frame mappings, reconstructed all crop geometry, verified
partial insertion bounds and ordinary edits, and losslessly recovered every
retained native float32 byte. It replayed **41,943,040 presence pixels** and
**1,342,177,280 hard bit predictions**, then re-encoded/re-decoded all **24 H264
clips** from retained pre-codec inputs. All decoded pixels and timestamps,
region/message/overlap readouts, source-level denominators, payload groups and
photo strata matched. It performed zero neural forwards or provider requests.

Native scoring completed in 114.53 seconds, sampled process-tree RSS
1,780,629,504 bytes, zero sampling errors. Replay completed in 13.15 seconds,
sampled RSS 303,988,736 bytes, with **one process-sampling error recorded**. The
monitor catches vanished/zombie-process events, but did not record the exact
exception type; it is not silently changed to zero. Both workers exited zero,
neither resource guard triggered, and sampled RSS remains a lower bound rather
than an exact peak-memory guarantee.

Seven model-free preparation/evidence tests passed before scoring. They cover
strict positive logits, the 499/500 component boundary, exact float-byte storage,
outside-interval versus marked-interval denominators, distinct original/derived
clocks and rejection of arbitrary external download routes. The selected static
and whitespace checks pass. The exporter verifies the original prospective
source revision and source-panel freeze before publishing
[complete aggregates and audit/profile receipts](evaluation-panels/public-visual-natural-v2-result.json).
Original retained records are unchanged; argument paths are withheld in public
resource-profile copies. The code, protocol, source/model hashes and aggregate
payload/photo-stratum results are public. Private photo/clip inputs remain local.

Before acquisition, 152 exact duplicate research files were converted to
byte-verified independent copy-on-write copies, recovering about 1 GiB of disk
space. All paths/content, permissions and timestamps were preserved; no hard
links, serving assets or customer uploads were involved. Native presence byte-
plane storage is also lossless. Resource savings did not reduce the case panel
or discard logits, messages, source frames or failures.

This is project-run development qualification with **zero outside human
reviewers**, zero customer inputs and zero external provider-verifier requests.
It supports further research into family localization while retaining false
responses and timing/message limitations. Genuine provider compatibility,
stronger false-alarm qualification and an admitted user-facing interpretation
are still required. No model, speech calibration, overall-confidence rule,
USD9.99 prepayment flow or GCP deployment changed.
