# Regional learning: source grouping and a differentiable prototype

The rejected v18 regional localizer needs stronger discrimination between an
inserted AI region and an ordinary edit. Changing its display gate alone did
not establish improvement. The next development route is supervised adaptation
of the backbone's small local/global layers together with its mask decoder,
using source-bound positive regions and matching non-generative controls.
It is not another fitted display rule over a frozen detector.

There is **no new project-image inference, training result, adoption or overall
confidence contribution** at this preparation stage. The first authored backward
probe and the actual source grouping are still pending. Existing serving checks
and rollback assets remain unchanged.

## Group related sources before choosing learning roles

The397 new source records contain320 attributed Flickr profiles. Metadata already
shows13 validation profiles and three testing profiles also in publisher
training; these are counts of shared profiles, not independent people verified
by us. Keeping the publisher role names alone would not prevent creator leakage.

The [prospective grouping freeze](evaluation-panels/image-source-groups-v22-freeze.json)
binds the complete397-source pair plan, every manifest covering the240 previously
consumed source originals, grouping code/tests, decoder versions and resource
monitor. After complete pair review:

1. Check all original byte and RGB commitments again. Preserve all397 new sources,
   including a source whose edit pair is explicitly unresolved; do not erase its
   possible links to other sources.
2. Compute a private256-bit horizontal-difference sketch of a17-by16 bilinearly
   resized grayscale thumbnail. There is no neural embedding or fitted cutoff.
3. Join sources with the same source ID, same attributed new-source profile,
   exact RGB/dimensions or a fixed near-thumbnail match. The latter requires at
   most8 differing bits,32–224 positive bits in each sketch, and aspect cross
   products differing by at most2% of the larger product.
4. Take the complete transitive components; order cannot split a chain of linked
   sources. Include all240 prior-consumed originals in the graph. Normalize
   integer/string source IDs so legacy formatting cannot hide a shared source.
5. Keep the complete graph and sketches private. Publish only counts and the
   whole-file commitment. Do not assign fitting, calibration, point, diagnostic
   or confirmation roles as a side effect of this audit.

This is a conservative leakage screen, not proof that every crop, burst, alias
or common photographer has been found. The prior240 sources do not yet have a
current photographer-profile check here. External checkpoint pretraining overlap
is unknown. A component containing a publisher nontraining source is therefore
a **candidate for later role review**, not automatically a fresh confirmation.
The earlier48-source confirmation minimum is not reduced by this preparation.

## Prototype to test before project fitting

The separate prototype strictly restores the complete commercially declared
Apache-2.0 image-only checkpoint and MIT source already inspected in v18. It
uses the native1024 canvas/patch/reconstruction graph, CPUfloat32 and two threads.
No generator weights, other checkpoints or implicit downloads are introduced.
Underlying checkpoint-photo clearance is not independently established by a
permissive weight card.

The authored probe selects the actually used local/global adaptation tensors
and decoder parameters, while freezing the dense base and unused paths. Eval
mode disables stochastic dropout while retaining differentiation. The prototype
exposes native pre-sigmoid logits so a confidently wrong prediction still
receives a stable corrective gradient. Its proposed loss gives equal weight to
positive and background pixels, adds a soft overlap term for edited examples,
and uses a negative-target mean loss for controls.

The pending full-model probe must establish exact parity with the released
inference graph, finite nonzero local/global/decoder gradients, unchanged dense
base parameters, and an exactly repeated one-step update after resetting the
same starting tensors. It uses a clearly authored mathematical target, not a
real/AI example. Passing it would demonstrate trainability only; it would not
establish detector quality or freeze a project training experiment.

All39 local authored preparation tests pass:14 grouping tests,20 parameter/loss
tests and five actual short-worker/reserve/refusal tests. The analytic loss tests
check known values and pixel-gradient directions, including a confidently wrong
negative example. The final report is committed by the grouping freeze; these
small tests are separate from the still-pending full-model backward probe.

## Execution and publication

Finish acquisition, primary qualification and the separate complete pair review
first. Run grouping and the authored neural probe sequentially under the same
900-second, sampled6-GiB and1536-MiB available RAM/disk policy. Preserve failed
attempts and sampling failures; do not overwrite them or automatically retry.
Resource use is sampled and does not guarantee an instantaneous peak.

Actual supervised learning still requires explicit connected-source roles,
source-bound recipes, fixed training/calibration/comparison rules, separate
diagnostics and a suitable untouched confirmation. Neither source grouping nor
successful backpropagation changes a customer score. Outside human reviewers
remain0. GCP remains deferred. GitHub Actions is disabled; check locally and
publish directly.
