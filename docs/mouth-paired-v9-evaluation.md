# Source-matched mouth learning: rejected after losing the prior gain

**Reject V9; retain V8 as the better research candidate.** V9 correctly located
**0/32 known mouth edits**, versus **14/32** for V8, with **0/56 ordinary
flags** for both. All 88 cases were assessed without execution failures.
The current production cue remains 0/32 locations and 1/56 ordinary flags on
this consumed panel. No candidate was enabled, no point was lowered, and no
fresh actors, customer confidence, payment or cloud settings changed.

## Exact method and source boundaries

The [predeclared design](mouth-paired-v9-development.md) reused all 1,600 V8
native stills and 5,440 verified prefix/view tensors: **16 fit actors**, **four
separate point actors**, and **eight excluded video-development actors**.
All are consumed sources/sentences, not new validation subjects. No new media,
weights or packages were downloaded. Existing permitted checkpoint declarations
and source ODbL/DbCL terms remain separate; training-rights/pretraining
disjointness are not established by this experiment.

Initialize from the exact V8 two-block learned state/head. Train only the last
two vision blocks, final norm and head (3,552,001 parameters), using a fixed
16-epoch schedule. Weighted BCE uses all fitting conditions. A matched ranking
term compares each first-family neural edit with the original, light and
resampling controls at the same source/frame, giving 768 fitting pairs.
A frozen V8 teacher constrains ordinary and second-family fitting responses,
using 4,096 eligible fitting-only view targets. No point actor enters either
term; evaluation needs only the one candidate model, without an original
reference or known generator. All frozen-prefix and teacher hashes stayed
unchanged; both declared trainable vision blocks changed.

Final training loss improved from .609684 to .419783; the matched-pair term
improved from .410315 to .157403. This did not yield useful video transfer.
The raw review point, **2.486741309111251**, came exclusively from ordinary
native margins of the four separate point actors, with a zero floor. The
fixed point, input sampling, window statistic and all old results remain
unchanged. Raw margins and points are not calibrated AI probabilities.

Review caught an inclusive-boundary mismatch in the standalone arithmetic
checker (`>` versus the detector's `>=`). The checker and post-hoc bound now
match the detector, with an exact-boundary regression. Original V8/V9 outcome
counts are unchanged; fitted points, weights and frozen scoring code did not
change. The earlier strict diagnostic is retained locally as superseded.

## Complete comparison

| Condition | AI edits | Current located | V8 located | V9 located |
|---|---:|---:|---:|---:|
| LivePortrait redraw | 8 | 0 | 0 | 0 |
| LivePortrait earlier retarget | 8 | 0 | 0 | 0 |
| MuseTalk aligned conditioning | 8 | 0 | 7 | 0 |
| MuseTalk earlier conditioning | 8 | 0 | 7 | 0 |
| **All AI edits** | **32** | **0** | **14** | **0** |
| **Ordinary controls flagged (56)** | — | **1** | **0** | **0** |

V9 left all 32 AI cases without flags, versus 18 for V8. V9 temporal/spatial
IoU means over all edits were zero, versus V8 .302009/.151177. Four of eleven
frozen gates passed (ordinary total/per-condition flags, assessed coverage,
complete replays). The seven gain/retention/location gates failed. Producing no
ordinary flags while missing every AI edit is not an accuracy improvement.
[All conditions, outcomes and gates](mouth-paired-v9-evaluation.json) are retained.

## Source and numerical verification

- All **88** current-baseline replays and V8-to-V9 native input comparisons
  passed. Learned features/margins may change; original RGB, PTS, alignment,
  native crop, mouth chip and normalized input identities may not.
- The [video source audit](evaluation-panels/mouth-v9-video-source-audit.json)
  independently decoded all **3,696** retained native inputs and replayed their
  crop/tensor/time bindings. The unchanged V8 audit format is reused with V9's
  lock/head/receipt identities; its `/8` format is not a claim of a V8 model.
- [Full-forward replay](evaluation-panels/mouth-v9-full-forward-replay.json)
  reloaded V9 and checked all **1,600** native stills. Full-model features and
  margins exactly matched retained-prefix inference. Point response arithmetic
  differed at most 8.88e-16; zero ordinary point stills crossed the fixed point.
- [Fitting replay](evaluation-panels/mouth-v9-fit-replay.json.gz) reproduced all
  **768 source pairs** and **4,096 teacher outputs**, including float32 payload
  and eligibility hashes. There were zero teacher offsets from point actors.
- The standard-library public reviewer reproduced **2,112 retained V8/V9
  windows**, all 88 clips across three arms, every pair's source/role/frame
  correspondence and all retained teacher-target/eligibility payload hashes.
  This does not certify neural inference, omitted windows, native masks,
  spatial IoU or generalization. **No outside reviewer has participated.**

All **101 focused research tests** passed, including held-out label/pixel
isolation, frozen-teacher/prefix preservation, same-source pair checks,
second-family retention gates, native-input parity and public-artifact tamper
checks. Software/source replay success does not override the failed detector.

## Diagnostic: a scalar point alone cannot rescue this response ranking

The frame-level point and the video's triplet/window means have different
distributions. All AI video-window means fell below the frozen frame point.
This motivates calibration at the actual unit of use, but that observation
alone does not establish a viable replacement.

A separate, explicitly [post-hoc ranking bound](evaluation-panels/mouth-v9-posthoc-ranking-bound.json)
uses only the consumed public responses. Any scalar point allowing at most
one ordinary flag must be strictly above the second-highest ordinary maximum,
**2.444695806585588**, because the detector uses `response >= point`. The
smallest permitted float64 point is **2.4446958065855884**. At that bound the
largest possible AI flag/location sets are **1/16 LivePortrait** and
**0/16 MuseTalk**. Every higher point can only shrink those sets. Thus even
oracle point selection on these consumed labels cannot meet the frozen gain
and retention requirements at the allowed ordinary-flag count. This is not a
new fitted point or population false-alarm estimate; correlated source variants
remain in their original denominators. No serving/research result was rewritten.

Next work should evaluate source-relative temporal evidence or a representation
that better separates edits from ordinary variation. Any new calibration must
use the actual video-window statistic with separate source groups. Preserve the
V8 gain while addressing the missed family; do not keep changing scalar cutoffs
on this panel. A passing development still needs unseen actors AND sentences,
multiple generators, realistic compression/resolution/ordinary edits, outside
review and customer evidence/paid-flow integration before local adoption.
Global-confidence admission and GCP remain separate, unfinished work.

Public [lock](evaluation-panels/mouth-v9-lock.json),
[learning/point receipt](evaluation-panels/mouth-v9-head.json),
[raw responses](evaluation-panels/mouth-v9-review-responses.json.gz) and
reproduction scripts contain no media, private workstation paths or learned
checkpoint binaries. The retained learned-state SHA is
`41cd916485c2f5b271a3932b1f8342a6169210a16b7771c5ce73ee53f5b23aa8`.
