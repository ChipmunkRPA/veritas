# Supervised regional image head: rejected, strict arithmetic failure retained

The learned regional head reduces false highlights on this development panel,
but finds fewer edits and overlaps their true changed pixels less well. It fails
the original continuation rules. **Do not replace the current image checks,
retune this result or select its successful generator subgroup for deployment.**
The current selector also remains weak; keeping it is not a claim of reliable
pixel attribution.

## Complete development comparison

The [protocol](image-patch-head-v14-protocol.md) and
[prospective freeze](evaluation-panels/image-patch-head-v14-freeze.json) were
published and remotely verified at
`479b7cca66dec2d8313f7cc977d0b212b6179477` before project-source scoring.
All 20 protected source/test/protocol files, asset identities and source roles
remain unchanged. The four self-authored native probes passed full-forward hooks
and one exact repeat: nine backbone forwards before any project-image scoring.

This uses all 288 consumed source groups from v13: 120 fitting, 60 point selection
and 108 diagnostics. It is source-disjoint between roles, **not fresh project
confirmation**. Every original/edit pair and ordinary/compression/lighting
variant stays in its role. TGIF/TGIF2's publisher declares CC BY-SA 4.0;
underlying photo rights and unknown pretrained overlap remain limitations.
Photos, pairings, 503,409,024 bytes of sampled float32 fit arrays and fitted
weights remain private. No customer inputs were used.

All 1,344 cases completed: 840 fit, 180 point-role and 324 diagnostic inputs.
The cached Apache-2.0, 86.6M-parameter backbone supplied frozen normalized tokens
from three layers. A 295,041-parameter head learned source-area edit supervision
from 53,760 sampled cells, with ordinary-edit and lighting/recompression controls.
It made 3,150 fixed optimizer updates, without early stopping or checkpoint
selection. The original trained weights and genuine-only point were frozen
before point-role AI inputs or diagnostic images were scored.

The point is an internal ranking rule, not a public screening threshold or AI
probability. Token attention supplies global context, so the resulting grid
cannot identify individual pixels' authorship. Each flag below means a nonempty
selected region; it can fail to overlap the actual edit.

| Diagnostic component | Edited files flagged /108 | Edits with IoU>=.3 /108 | Mean edited IoU | Genuine flags /108 | Ordinary-edit flags /108 |
|---|---:|---:|---:|---:|---:|
| Current input-bound selector | 19 | 6 | .043006 | 7 | Not assessed |
| Learned regional head | 10 | 3 | .017772 | 0 | 0 |

Mean pixel F1 falls from .064904 current to .027452 candidate, including empty
masks as misses. Lower false highlights do not offset losing useful detection.
The candidate passes the two false-highlight rules, but fails higher mean IoU
and more IoU>=.3 cases. No missed input or failure was dropped.

| Generator,36 edited/genuine/ordinary inputs each | Current edited flags / IoU>=.3 / genuine flags | Candidate edited flags / IoU>=.3 / genuine / ordinary flags |
|---|---:|---:|
| Firefly | 5 / 2 / 2 | 3 / 1 / 0 / 0 |
| SD2 | 11 / 4 / 3 | 3 / 0 / 0 / 0 |
| FLUX Fill | 3 / 0 / 2 | 4 / 2 / 0 / 0 |

The FLUX subgroup improves, but the aggregate rule fails and all subgroup results
remain diagnostic. On the 60-source v11 diagnostic cohort, mean IoU falls from
.038637 to .019195; on the 48 consumed v12 sources it falls from .048467 to
.015993. The outcome does not reverse between prior-source cohorts.

Fitting loss decreases from .279093 to .079198 while held-role localization
remains weak. Low fitting loss cannot stand in for useful detection. This study
does not isolate whether representation, source variation, ranking-point design
or spatial granularity causes the quality loss. No unused fresh confirmation
panel or current-selector ordinary-control inference was spent after failure.

## Original strict audit and separate diagnostics

The original strict auditor exited 1 after its float32-versus-float64 full-image
mean check exceeded the prospectively fixed 1e-5 bound. It ran for 4.47 seconds,
with no resource guard or sampling error. No complete original `audit.json`
was created. Its code, failed log and profile are retained, and the bound is
unchanged. The partial audit did not instrument completed backbone forwards;
its exact count is not invented from elapsed time.

The [separate diagnostic plan](image-patch-head-v14-diagnostic-plan.md) and
[failure-bound freeze](evaluation-panels/image-patch-head-v14-diagnostic-freeze.json)
were published and remotely verified at
`498023d79f83570702694ce5348cd63c7d5ffede` before a new diagnostic forward.
This investigation records every original mean-bound failure while keeping the
original trained head, point, source panel and all other numerical checks fixed.
It cannot convert the failed strict audit or rejected quality result into a pass.

NumPy's [version-matched mean documentation](https://numpy.org/doc/2.2/reference/generated/numpy.mean.html)
explains that float32 inputs use float32 accumulation by default and that a
higher-precision accumulator can reduce error. That general explanation does
not waive this experiment's bound or qualify an untested corrected model.

The separate diagnostic replay completed all **1,344 cases / 1,362 backbone
forwards**, including 18 full-forward hook probes. It reproduced all
3,170,893,824 native feature values exactly, every sampled fit vector/target,
all 516,096 retained response cells, source masks, the genuine-only point and
all outcome aggregates. Independent integral pooling and component/rectangle
projection checked 939,015,007 source pixels including repeated variants.
The separate float64 dense-head arithmetic differs by at most 3.67261e-6,
within its unchanged 3e-5 bound. Six response cells lie within that tolerance
of the point; exact original/native grids govern the original result.

The original mean bound fails in **1,302/1,344 cases**: 813/840 fit, 174/180
point-role and 315/324 diagnostic cases. The largest difference is
7.69600e-5; the first replayed failure is at index zero. These are repeated
conditions, not independent population samples. The native float32 means
themselves reproduce exactly; this is a precision-validation failure, not
evidence that source images or saved outputs changed.

With the original trained weights and point fixed, comparing two float64 head
paths using the original versus accurately accumulated mean changes responses
by at most 9.69410e-7 and changes **zero of 504 selected source masks**.
This isolates inference sensitivity. It neither retrains with corrected context
features nor creates a qualified alternative model. The weaker localization
result is not repaired by this mean-input probe. The original strict audit
remains failed.

Diagnostic replay took 445.20 seconds with 1,427,275,776 bytes sampled tree RSS,
zero sampling errors and no guard. The full
[aggregate publication](evaluation-panels/image-patch-head-v14-result.json)
preserves both completed phases and the failed original audit, including
unaltered error/exit counts. Its 2,725 completed scoring/diagnostic backbone
forwards exclude the original strict auditor's uninstrumented partial count;
the nine authored-contract forwards are also reported separately.

## Separate-host synthetic checks

The [actual Linux run](https://github.com/ChipmunkRPA/veritas/actions/runs/42474888478)
passed all 11 synthetic checks on Ubuntu/x86_64, Python 3.12.6 and Torch 2.13.0+cpu.
It checked the 20 source commitments from the immutable prospective revision,
authored RGB normalization, odd-size soft-label geometry, role isolation,
toy-head learning, independent dense/ReLU/sigmoid/median arithmetic and exact
component projection. Network calls and pretrained-backbone loading were
forbidden. The [decoded-log receipt](evaluation-panels/image-patch-head-v14-linux-synthetic-review.json)
records the actual runtime and scope.

This run uses no real research images or pretrained weights. It does not repeat
the learned real-image head or its quality comparison, and does not repair the
separate native mean failure. Two further local diagnostic tests preserve that
failure and refuse any claim that diagnostic completion authorizes admission.
There are zero outside human reviewers.

The main native run completed 1,363 backbone forwards in 460.01 seconds, with
1,413,398,528 bytes sampled process-tree RSS, no guard and zero sampling errors.
This is a sampled lower bound rather than an exact memory peak. Both the current
serving weights and rollback assets remain. The speech calibration, USD9.99
prepayment rule, customer reports and GCP deployment are unchanged.

A private local preview is generated from exact bound sources and selected masks for the strongest
overlap, a weak-overlap flag, a large missed edit and a genuine control. It is
kept outside GitHub along with the raw research photos; the plotting code is
public. No candidate region or score is admitted into customer reports.
