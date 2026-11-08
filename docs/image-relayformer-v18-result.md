# Regional localizer v18: replacement rejected

**Keep the current image selectors.** The new localizer/readout finds0/48
useful AI-edit regions in native images and0/48 in JPEG75, versus9/48 and6/48
current. It loses every previously useful location. Native originals also
exceed the false-highlight budget. The candidate does not qualify for fresh
confirmation, default replacement or a customer-confidence contribution.

All384 planned inputs completed. Every full source float32 map then repeated
exactly in the complete local replay; independently reconstructed components,
rankings, pixel counts and both genuine-only points match. This is a
reproducible negative quality result, not evidence that a successful runtime
probe establishes detection accuracy.

## Fixed comparison and outcomes

The [protocol](image-relayformer-v18-protocol.md), complete execution/review
code and [freeze](evaluation-panels/relayformer-v18-freeze.json) were published
as `59e544bbbc2f4aff6a9611fd158ae30dcac1817f`. The remote freeze blob
`50137d92e0418e8432bc311830590890259619d3` was verified before the first
project-image forward. Private freeze SHA256:
`2d2614e84f22eb716d63b8e2d88ebce6a78fa30433120cee72153597cb9677d6`.
All protected files and original rules remain unchanged.

The study reuses consumed TGIF/TGIF2 sources:24 genuine point groups and48
source-disjoint diagnostic groups across Firefly,SD2,FLUX Fill. Native/JPEG
and original/ordinary/AI variants are correlated observations. The96 point
inputs contain no AI image. Their complete journals and per-format exact
rational gates were sealed before all288 diagnostics.

Each format contains48 edited,48 original and48 ordinary retouch images.
Useful means source-mask IoU>=.3; empty/missed masks remain in the denominator.
The comparison uses retained input-bound current-selector outputs, not a new
full SaaS run. Publisher masks are not independent pixel-authorship proof.
Here an original means the publisher's original/control, not an independently
authenticated raw-camera capture. Prior ordinary editing or artistic content
can exist in a source photograph; flags are errors against the benchmark's
AI-edit target, not proof that no earlier manual editing occurred.

| Measure | Native current | Native candidate | JPEG75 current | JPEG75 candidate |
| --- | ---: | ---: | ---: | ---: |
| Useful regions, /48 edits | 9 | 0 | 6 | 0 |
| Any edit overlap, /48 | 10 | 2 | 9 | 1 |
| Mean edited IoU | .149765 | .000534 | .079315 | .000409 |
| Mean edited pixel F1 | .168340 | .001053 | .107395 | .000802 |
| Originals highlighted, /48 | 0 | 2 | 5 | 1 |
| Ordinary retouches highlighted, /48 | 33 | 8 | 23 | 8 |
| Mean original area wrongly highlighted | 0% | 2.968% | .371% | 1.666% |
| Mean ordinary area highlighted | 4.664% | 6.120% | 3.820% | 4.534% |

Native fails four rules: more useful regions, higher mean IoU, preservation
of current useful locations and the original-image highlight budget. JPEG
fails the first three. Fewer ordinary alerts do not offset losing localization,
and **fewer alerts can conceal larger wrong regions**. The single flagged JPEG
original covers about80% of that image, whereas the current selector's five
original flags are much smaller. Both alert counts and affected area matter.

Paired48-source bootstrap95% intervals for mean-IoU differences are
[-.243106,-.066605] native and[-.133307,-.032204] JPEG. The source groups and
sealed points remain fixed. These are development-cohort intervals, not
customer confidence intervals, a public threshold or population reliability.
Complete source-family diagnostics and every rule are in the
[aggregate package](evaluation-panels/relayformer-v18-result.json).

## Why changing the gate does not establish a replacement

The area gate uses actual retained components, not an AI-probability score.
Broad false regions on genuine point controls result in strict lower bounds
of168503/786432 source area for native and107609/837837 for JPEG, in addition
to the fixed1/1000 floor. Those settings are unchanged after diagnostics.

A post-result descriptive check of the already retained ungated proposals finds
11/48 useful native edits and6/48 JPEG edits. It also highlights38/48 and45/48
untouched originals, and43/48 and48/48 ordinary retouches. These are not new
passing arms or fresh results. The gate suppresses useful proposals along with
false ones, while the underlying unqualified proposals are also far too broad
on genuine inputs. Lowering a gate is not demonstrated useful detection at a
similar false-alarm rate.

The tested complete method is unsuitable as a replacement on this panel. This
does not prove the architecture universally unusable or reproduce its paper's
different training/test datasets. Further work needs stronger regional
discrimination and verified AI-specific targets/ordinary controls. Neither
raw sigmoid maps nor edit-area fractions can become AI-origin confidence by
renaming them.

## Runtime, weights and complete local review

The [permission/structural preparation](new-localizers-and-editing-methods-2026-11-08.md)
records separate MIT source and official Apache-2.0 checkpoint declarations,
weights-only loading and exact matching of all410 released tensors. The full
checkpoint is restored without timm downloads, unsafe base-loader fallback or
initialized substitutes. Six authored native forwards preceded the source
freeze. Canonical study execution uses CPUfloat32, two threads and unmerged LoRA.

Every input's actual patch tensor is bound to its source rectangle. Full maps
are retained losslessly as float32; no8-bit map quantization substitutes for
them. Components use exact float32-unit sums and rational mean ranking, with
integer source-center projection and exact source-area gates. The source's
oversized-image crop behavior is not silently inherited. New video inference,
video-tail handling and video checkpoint quality remain untested.

| Phase | Inputs | Actual localizer forwards | Resource-guard duration | Sampled tree RSS lower bound |
| --- | ---: | ---: | ---: | ---: |
| Source/reference freeze | 384 reconstructed | 0 | 22.871s | 690,946,048 bytes |
| Genuine-only points | 96 | 108 | 181.021s | 1,340,309,504 bytes |
| Diagnostics | 288 | 306 | 493.094s | 1,338,638,336 bytes |
| Full native/source replay | 384 | 384 | 610.119s | 1,393,459,200 bytes |

All phases exit0, without guard or sampling error, inside900seconds/6GiB and
the1536MiB RAM/disk reserve. RSS is sampled, not a true peak guarantee.
The study completes **798 project-source forwards**, including30 original
context repeats and384 replay forwards. The six earlier authored localizer
forwards are separate; no new current-selector neural run is claimed.

The full reviewer re-infers every map bit and input/patch proof. OpenCV/IEEE
mantissa arithmetic separately reconstructs all384 retained component masks
and ranks; independent contingency tables recount all288 gated diagnostic masks.
Both point refits and every summary/decision match. The current renderer uses
its retained canonical path; it is not a new independent implementation. This
local review uses the same neural libraries and is not outside human review.
The historical strict IML rounding audit remains failed and is not cleared.

All44 authored tests pass across preparation and the frozen study:32 coverage,
projection and missing-weight tests, plus12 exact-ranking/genuine-role/count
checks. The private visual review includes a strong ungated proposal, a large
miss, a genuine false region and ordinary retouch, selected only to illustrate
failure categories after results. It is not representative-rate evidence.
Underlying photos, full maps, source pairings and the private preview are not
published to GitHub. The plotting and review code are public.

## Remaining work

Do not spend new confirmation sources on this failed method or select a
successful subgroup. Keep current model assets and rollback; the image baseline
also remains weak. Sources are consumed, external training overlap and underlying
photograph rights are unverified, and no production-training grant is inferred
from this evaluation. A training-data rights audit and targeted regional-method
development remain important next work before another confirmation.

Source credit: Hannes Mareen and colleagues, *TGIF: Text-Guided Inpainting
Forgery Dataset* (WIFS2024), and *TGIF2: Extended Text-Guided Inpainting Forgery
Dataset & Benchmark* (2026). The
[IDLabMedia release atfc5c557](https://github.com/IDLabMedia/tgif-dataset/tree/fc5c557bd1a2d49a5dabc3676a54cfc44798f4ea)
declares CC BY-SA4.0. Our ordinary retouch/JPEG recipes and source projections
are modifications; raw photographs remain private. Its blanket statement about
COCO image licensing does not independently establish each underlying photo's
rights for production training.

Outside human reviewers remain0; the repository has no submitted evaluation
issues at the observed check. Public review instructions do not imply recruited
participants. Local checks and direct pushes continue with Actions disabled.
Serving confidence, customer model-name filtering, USD9.99 prepayment and GCP
deferral are unchanged. Image edits, difficult audio, lip sync/general video,
writing and cross-media calibration remain unfinished parts of the full goal.
