# Image context precision and region calibration: development result

The predetermined primary candidate **failed development and remains disabled**.
Accurate context means passed the original numerical requirement, and region
calibration improved localization relative to the rejected old head. The primary
still localized fewer edits than the current selector. No subgroup or ablation
was substituted and no fresh confirmation data were consumed.

The [protocol](image-region-point-v15-protocol.md), 30 protected source files and
[panel freeze](evaluation-panels/image-region-point-v15-freeze.json) were published
in commit [730e11e](https://github.com/ChipmunkRPA/veritas/commit/730e11e75e839c8f4650ec45d6638e6d48bad9d5)
and verified remotely before new project scoring. The
[complete aggregate result](evaluation-panels/image-region-point-v15-result.json)
retains every arm, generator/cohort, point, audit and resource measurement.
This is consumed development data, not fresh accuracy or a watermark study.

## Complete outcomes

Each arm has 108 edited, 108 genuine and 108 ordinary-adjustment diagnostic
inputs. Intersection over union (IoU) measures displayed-mask overlap with known
edits; IoU >= .3 is the frozen useful-localization criterion. Mean IoU and pixel
F1 include misses as zero. These are localization metrics, not AI confidence.
An empty mask cannot establish originality.

| Method | Edits flagged | Useful overlap | Mean IoU | Mean pixel F1 | Genuine flags | Ordinary flags |
|---|---|---|---|---|---|---|
| Current selector | 19/108 | 6/108 | .043006 | .064904 | 7/108 | Unassessed |
| Old head / maximum | 10/108 | 3/108 | .017772 | .027452 | 0/108 | 0/108 |
| Old head / regions | 16/108 | 4/108 | .026576 | .040549 | 3/108 | 1/108 |
| Stable means / maximum | 12/108 | 3/108 | .021815 | .033236 | 0/108 | 0/108 |
| Stable means / regions — primary | 14/108 | 4/108 | .028234 | .042905 | 2/108 | 1/108 |

The primary missed 94/108 edits entirely; 104/108 did not reach useful overlap.
Current missed 89/108 and 102/108 did not reach useful overlap. Reducing genuine
false highlights from seven to two cannot compensate for failing both frozen
localization improvements. Current-selector ordinary controls were deferred
prospectively until development passed; that comparison is unassessed, not zero.
All four arms fail the pooled quality requirements.

| Generator / method | Edits flagged | Useful overlap | Mean IoU | Genuine flags | Ordinary flags |
|---|---|---|---|---|---|
| Firefly / current | 5/36 | 2/36 | .036630 | 2/36 | Unassessed |
| Firefly / primary | 6/36 | 1/36 | .024006 | 0/36 | 0/36 |
| SD2 / current | 11/36 | 4/36 | .078833 | 3/36 | Unassessed |
| SD2 / primary | 3/36 | 1/36 | .017349 | 0/36 | 0/36 |
| FLUX-Fill / current | 3/36 | 0/36 | .013555 | 2/36 | Unassessed |
| FLUX-Fill / primary | 5/36 | 2/36 | .043346 | 2/36 | 1/36 |

The FLUX-Fill gain cannot replace the predetermined pooled decision. Mean IoU
also fell from .038637 to .028330 on the 60 v11 diagnostic edits, and .048467 to
.028115 on 48 previously consumed v12 edits. JSON retains all four arms in
both cohorts and all families. These are correlated cases, not independent votes.

## Four fixed points and retained precision failure

The same Apache-2.0 register-base backbone and 295,041-parameter context head
were retained. The new means accumulate 1,024 float32 patch vectors in float64
and round once. Refit uses the same seed, 30 epochs, 3,150 optimizer updates,
53,760 sampled patches and source-area targets. Training loss fell from .279093
to .079449, without selecting a checkpoint on diagnostic outcomes. Fitting loss
does not establish detection quality. Existing vectors/targets are reused
read-only; another 503 MB copy was not made.

The region point represents the highest level with an 8-connected component
of at least four cells. Isolated maxima need not survive that display rule.
Both heads and all four points were frozen after 840 fit cases and 120 genuine
point cases, before new AI point cases or diagnostic scoring. All points use
the same 60 genuine source groups, original/ordinary variants, descending rank
three and strict greater-than comparison. No AI example selects a point.

| Arm | Internal ranking point | Point groups with displayed regions |
|---|---|---|
| Old maximum | .9993374943733215 | 0/60 |
| Old regions | .9978950023651123 | 3/60 |
| Stable maximum | .9992156028747559 | 1/60 |
| Stable regions | .9987490177154541 | 3/60 |

These are ranking points, not AI probabilities or public screening thresholds.
Three in-sample groups is not a population false-alarm guarantee. The old-head
arms retain their failed original mean criterion and cannot become serving
candidates through this experiment. Original v14 outcomes, arrays, head, failed
strict audit and subsequent diagnostics remain unchanged.

## Complete native replay and actual Linux arithmetic check

Development completed all 1,344 cases, 18 full-forward hooks and one exact
repeat: 1,363 backbone forwards. The separate audit completed all 1,344 cases
plus 18 hooks: another 1,362 forwards. It replayed 3,170,893,824 native feature
values, all 840 fit rows / 53,760 sampled vectors, both heads' 1,008 response
grids / 1,032,192 cells, all 2,016 source-mask outcomes and source projection
covering 939,015,007 pixels including repeated variants. Independent binary
search/flood fill reproduces activation levels and all four points, without
reusing the primary union algorithm.

The largest new-mean difference from independently grouped float64 arithmetic
was **3.727735e-6**, within the unchanged **1e-5** bound. All **1,302** old mean
failures reproduced and remain failures. Independent dense/ReLU/sigmoid/median
arithmetic differs by at most **5.130478e-6**, within **3e-5**. Native features,
maps, fit samples, source metrics and points match exactly. The optimizer was
not independently refitted; outside human reviewers remain **zero**.

A real [Linux run](https://github.com/ChipmunkRPA/veritas/actions/runs/42478007916)
passed **12 array-only tests**, with the
[actual receipt](evaluation-panels/image-region-point-v15-linux-array-review.json).
It checks source commitments, stable means, all 729 small plateau fields,
component activation, native-selection equivalence and point-role isolation.
It loads no Torch, models, project images or vectors; network is blocked during
tests. This validates portable arithmetic, not native image inference or outside
human review. Workflow commit d089e72 differs from science commit 730e11e.

Native scoring took **460.793 s**, sampled process-tree RSS at least
**1,401,896,960 bytes**. Full audit took **454.168 s**, at least
**1,438,695,424 bytes**. Both exited zero with no guard and zero sampling errors,
inside their separate 900-second / 6-GiB limits. The authored four-image native
contract completed nine forwards before scoring. Sampled RSS is a lower bound.

## Consequence

Keep the current selector and rollback assets. The new arithmetic and component
point are verified engineering changes; this head remains too weak at edit
localization to adopt. A subsequent method needs a new frozen development
comparison, then untouched sources, matched ordinary controls and full-app/export
verification after passing development. Raw photos, source pairings, vectors,
learned private weights and customer files were not published. Speech calibration,
billing, external customer-verifier settings and GCP resources were unchanged.
