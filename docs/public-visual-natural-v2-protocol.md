# Natural-media public watermark qualification, fixed before scoring

This is a prospective qualification of the released MIT WAM watermark family
on natural-photo and real-camera clip inputs. It extends the completed
[authored contract](public-visual-watermark-v1-result.md), retaining its native
preprocessing, presence-logit point and component rule. It is not a general
AI-image, face-swap, lip-sync, audio or text detector benchmark. A public mark
can be applied to genuine material and cannot authenticate its provider or AI
authorship. This study cannot admit a confidence contribution or fast-path stop.

## Sources and source review before inference

The retained metadata preflight contains 288 distinct Wikimedia file pages in
16 API responses. Per-file MIME, dated upload/capture claims, credited creator,
license name and license URL produce 78 photo and 24 clip candidates. Accepted
license routes are the selected CC BY / CC BY-SA versions and CC0; no NC/ND
weight or source permission is substituted. Final selected sources all carry
their individual attribution/terms in the private source ledger. Publisher
metadata is a claim, not a cryptographic certificate of capture or ownership.
The selected panel has 27 CC BY-SA 4.0, 11 CC BY-SA 3.0 and two CC BY 3.0
sources. The primary [BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/),
[BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and
[BY 3.0](https://creativecommons.org/licenses/by/3.0/) terms permit commercial
sharing/adaptation with their attribution and applicable ShareAlike conditions.
Individual credits and change records are retained; these terms do not certify
capture truth or supply every possible non-copyright permission.

Select **32 photos** prospectively: ten architecture, ten food, six plants and
six landscapes from the retained category pages. No more than two photos per
credited creator; actual selection has 25 credited photo creators. Select
**eight short clips** from the river/waterfall category pools, with eight
distinct credited creators. Rank by a fixed hash of page ID, not image content
or detector responses. File/capture metadata must indicate dates before 2021;
clip duration/derivative-size bounds are fixed in the preparer. Known described
animations, diagrams, illustrations and violent/military footage are excluded.

Search all project JSON/JSONL inventories for earlier Wikimedia URLs, retaining
matching inventory checksums and excluding earlier source URLs. This run finds
no earlier matching project inventory outside its own source-preflight folder.
This is freshness to known project URL inventories, not a claim about model
pretraining overlap, independently verified capture truth, or all possible
perceptual duplicates. Byte/RGB identities of downloaded and working media are
retained. No-added-mark controls mean this study adds no mark, not certified
original/human files or guaranteed absence of an unknown prior mark.

Acquire the declared thumbnails and 240p VP9 transcodes, preserving downloaded
bytes, full per-file metadata/credits, working RGB arrays, exact checksums and
crop/clock mappings. Review all 40 sources through a 56-preview contact sheet
before scoring: all 32 photos and first/middle/last sampled frames of every
clip. This is development visual eligibility review, not independent human
authorship verification. All selected previews were photographically eligible;
no selected source was replaced after seeing a detector response.

## Fixed cases and native readout

- Work on a centered 256×256 bilinear crop, recording the crop in the downloaded
  derivative's coordinates. No full-resolution original was examined; do not
  pretend its precise pixels were checked.
- Each photo has no-added-mark, fully marked, center-128-square partially marked
  and conventionally edited center-crop states, each native and JPEG quality 70:
  **256 photo cases**. The ordinary edit rotates the same center crop 90 degrees.
- Each clip has eight distinct sampled frames, nearest original PTS to 0.5-second
  offsets from its first frame. Preserve each original frame index, PTS/time base
  and timestamp. Derive a separate 2fps sequence; never replace the original
  timestamps with idealized derived times.
- No-added-mark, partial-mark and ordinary-edit clip sequences, each as raw
  frames and H264 CRF 23 / YUV420p: **384 frame cases**. Partial and ordinary edits
  affect the center square only on sampled frames 2–5 inclusive. Derived 1–3s
  maps to the recorded source samples. Unsampled source frames were not checked.
  Outside-interval codec effects cannot automatically be called new AI edits.
- Four predeclared 32-bit messages—alternating, zeros, ones and fixed SHA-derived
  bits—are assigned by numeric source ID modulo four. They test payload diversity
  without changing detector weights, preprocessing, thresholds or decision rules.
- Use the same publisher ImageNet transform, native F32 generation, scaling_w2,
  JND and half-up RGB8 quantization. Presence is strictly first-channel logit>0;
  8-connected components of at least 500 pixels flag inspection. Keep all
  selected pixels, including smaller components. This point is an inspection
  rule, not a validated false-alarm guarantee or public AI-confidence threshold.
- Decode all 32 hard bit maps over the selected mask. Empty selection is
  unavailable. Check against the actual assigned message and publisher hard
  readout. Neither a bit-match proportion nor a family-presence flag is an
  AI-authorship confidence score.

## Retention, resource limits and independent replay

Retain every native presence float32 byte losslessly as transposed byte planes,
with original F32-byte SHA-256, every packed hard bit map and transformed input
RGB. This changes compression only; no rounding or float precision is removed.
Keep original assets, working arrays, all source/marked/partial/ordinary base
arrays, every H264 clip and an append-only per-case journal privately. Raw
continuous bit logits remain unused; the hard maps preserve the actual readout.

Run **64 generator and 645 detector forwards**, including four first-photo
wrapper/direct comparisons and an exact repeated partial prediction: **709
neural forwards** total. No hidden repeat, score-dependent replacement, dropped
failure or threshold change is allowed. CPU F32, two Torch threads, seed
20261108; guard 900 seconds / 6 GiB sampled process-tree RSS / 1536 MiB available
RAM and disk. A guarded or failed job remains recorded; no implicit restart.

A separate non-neural audit must independently re-decode all downloaded source
assets, reconstruct crop geometry and original sample clocks, check exact source
RGB, insertion bounds and ordinary edits, recover every original presence byte,
recompute masks/components/messages/overlaps and all group denominators. It
re-encodes every H264 clip from retained pre-codec inputs and requires identical
decoded frames and time metadata. Repeated container bytes are not an extra
acceptance condition; exact decoded pixels and timestamps are. The audit has
the same resource guard and performs zero neural inference.

Report every source/state/codec group, payload and photo stratum. Include source-
level any-frame flags and complete marked-interval misses, marked-frame misses,
flags outside intended intervals, mean IoU, flagged marked cases with IoU≥0.5,
exact message recovery and unavailable-message counts. IoU≥0.5 is a localization
summary, not a change to the inspection rule. Frames/variants share sources;
do not represent them as independent population samples or invent confidence.

The prospective freeze binds source plan/acquisition/visual-review checksums,
all study/helper/auditor/test/protocol code, the original authored dependencies,
exact packages and pinned public model/source/dependency identities. Publish
and verify code/freeze before native scoring. Source assets, case pairings and
prediction arrays remain private; methods, hashes, terms summaries and aggregates
may be public. No customer inputs, provider-verifier requests or outside human
reviewers are involved. Serving models, calibration, USD9.99 prepayment and GCP
deployment are unchanged.
