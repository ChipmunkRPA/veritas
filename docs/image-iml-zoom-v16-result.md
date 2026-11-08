# Magnified image crops: replacement rejected

Magnifying overlapping crops did **not** improve useful edit localization.
The crop candidate found **0/48 useful edited regions in native images and
0/48 in JPEG75**, versus **9/48 and 6/48** for the current format-specific
checks. Both formats fail their frozen quality requirements. Keep the current
checks and customer confidence unchanged; do not spend fresh confirmation on
this candidate.

All 384 planned candidate inputs and 96 new ordinary-reference inputs
completed without a failed image. The original strict independent-mask audit
**failed** on a rounding-boundary discrepancy and remains failed. A separately
published post-result audit reconstructs all source inputs and canonical masks,
quantifies numerical sensitivity, and confirms that rejection does not change.
That supplement does not clear the original exact-reconstruction failure.

## Matched outcome

These are consumed-source development results, not fresh accuracy. Each format
has 48 source triplets: original, known AI edit and ordinary central blur/tone
retouch. A useful location has displayed-mask IoU >= .3 with the known edit
mask. Misses remain in the denominator with zero overlap. Highlights on ordinary
retouches are reported separately from highlights on original photographs;
neither count is a whole-file AI verdict.

| Format | Method | Useful locations /48 | Any edit overlap /48 | Mean edit IoU | Original highlights /48 | Ordinary highlights /48 |
|---|---|---:|---:|---:|---:|---:|
| Native | Current check |9|10|.149765|0|33|
| Native | Matched whole-image control |1|2|.009942|2|5|
| Native | Magnified crops |0|1|.001162|1|4|
| JPEG75 | Current check |6|9|.079315|5|23|
| JPEG75 | Matched whole-image control |0|0|0|2|7|
| JPEG75 | Magnified crops |0|1|.000353|2|5|

The crop method reduces ordinary-retouch highlights relative to the current
checks, but loses all current useful edit locations. Native crops also lose
the matched whole-image control's one useful location. JPEG's tiny average
overlap gain over its zero-overlap control supplies no useful location and
does not outperform the current check. Lower false highlights alone are not
sufficient useful detection.

The native format fails five rules: more useful locations and higher mean IoU
against both references, and preservation of current useful locations. JPEG
fails four: more useful locations against both references, higher mean IoU
against current, and preservation of current useful locations. Coverage and
original/ordinary-highlight budgets pass. No successful generator or new point
was selected after results. All family summaries and every rule remain in the
[aggregate result](evaluation-panels/image-iml-zoom-v16-result.json).

Paired-source mean-IoU differences versus current are -.148603 for native
(bootstrap 95% percentile interval [-.241057, -.067095]) and -.078962 for JPEG
([-.134082, -.032250]). These resample 48 consumed diagnostic sources while
holding the genuine-only points fixed. They exclude point-selection uncertainty
and are not customer confidence intervals or fresh-population accuracy.

## Frozen inputs, method and permissions

The [protocol](image-iml-zoom-v16-protocol.md) and
[input/code freeze](evaluation-panels/image-iml-zoom-v16-freeze.json) were
published as [d1165ff](https://github.com/ChipmunkRPA/veritas/commit/d1165ffe510eabb38355d0511e52388bea0097a1)
before new model responses. The two complete genuine-only journals and all
format/arm selection points were sealed before diagnostic processing. The
additional independent count auditor was published as
[910a950](https://github.com/ChipmunkRPA/veritas/commit/910a9502179f625d7861a23216bc8e84eab99ac4)
and remotely verified before diagnostic scoring.

There are 24 genuine point sources and 48 source-disjoint consumed diagnostics
across Firefly, SD2 and FLUX Fill. Variants are correlated: 384 candidate files
come from 72 source groups, not 384 independent photographs. Point sources
supply original/ordinary inputs in two formats; diagnostic sources additionally
supply their known AI edit. Source masks retain original edit coordinates;
JPEG changes do not become AI truth.

The method uses the existing permitted IML-ViT checkpoint and independent
compatible head. The checkpoint owner explicitly permits commercial use in
[issue60](https://github.com/SunnyHaze/IML-ViT/issues/60). Source, checkpoint and
data rights remain separate. TGIF/TGIF2 publisher attribution/CC-BY-SA4.0 and
underlying photograph rights are retained. No restricted decoder, new model
download, training or customer upload was used.

Four overlapping corner crops cover 60% of each dimension, each magnified to
long side1024. Source-edge coverage and internal-border exclusions are fixed.
The candidate averages tile evidence and combines it equally with the full map.
The matched whole control shares the same coarse readout. Five lossless
256x256 uint8 grids are recorded privately per input. This was predeclared;
the study does not retain every full-resolution new model output. Native crop
inputs are checked at the actual model call. Each of eight phases repeats its
first whole prediction and crop, including old whole-map/grid parity.

All **1704 actual candidate forwards** and **98 actual current-reference
forwards** succeeded. Whole original/AI maps are otherwise reused under their
source/map commitments. All 96 ordinary-reference maps are newly inferred.
These are inspection maps, not calibrated pixel-authorship probabilities.

## Failed strict audit and complete supplementary diagnosis

The frozen strict reviewer stopped on an ordinary genuine point control:
its independently reconstructed map differed by 2.220446049250313e-16, but
rounding changed the displayed false-positive count from 6074 to 6075. Its
original source, five-second failed profile and log remain unchanged. No
complete strict-review receipt was produced and none is fabricated.

The [supplementary protocol](image-iml-zoom-v16-boundary-audit-protocol.md) and
[new diagnostic freeze](evaluation-panels/image-iml-zoom-v16-boundary-audit-freeze.json)
were published as [2157287](https://github.com/ChipmunkRPA/veritas/commit/215728702b91df9b1afe52968ab1b42c015eb7c0)
before its full replay. This was explicitly post-result numerical diagnosis.
It changed no original map, point, outcome, rule or failed audit.

The complete supplementary run checks all **384 source cases /1920 grids**,
independent ordinary/JPEG input reconstruction, current-reference masks and
**864 canonical diagnostic arm metrics**. Independent connected components
and pixel counts on the exact canonical maps reproduce the stored results.
Separate count/gate/bootstrap arithmetic also matches all original outcomes.
Inline native-input/reduction checks are inherited; neural outputs are not
all re-executed by these audits.

Equivalent continuous reconstruction differs by at most
**4.440892098500626e-16**, within the unchanged 1e-12 bound. Nevertheless,
325646 quantized cells across all map comparisons choose neighboring 8-bit
levels at rounding discontinuities. Many are below the selection cutoff, but
**25/768 ungated map metrics differ**; component connectivity/filtering can
amplify a changed boundary pixel. The largest positive-area difference is
1206 pixels. The native crop genuine-only point changes from
.18790414442194406 to .18789839147643597; the other three points match exactly.
Thus exact independent discrete reconstruction and point parity remain failed.

Sensitivity using the independent maps with either the original sealed points
or the equivalently reconstructed genuine-only points still has **zero useful
crop locations in both formats**. Original/ordinary highlight counts and all
advancement decisions remain unchanged. Small overlap values change; neither
alternative is selected as a new policy. The public JSON retains both summaries
and discrepancy counts by role, format, operation and arm.

## Runtime and public reproduction

All native phases run sequentially under the declared 900-second/6-GiB sampled
process-tree guard and minimum1536-MiB available RAM/disk. The eight candidate
batches take about223–233 seconds each, including startup, four crop forwards,
recording and inline checks. Reference phases take15.766/42.258 seconds; summary
18.676; failed strict review5.024; supplementary review69.659. These are research
phase runtimes, not a deployed service-latency benchmark.

Maximum sampled process-tree RSS is1,382,973,440 bytes. One sampling error in the
second family's native phase remains recorded. RSS is a lower bound on peak use;
it is not a complete memory trace. No resource guard interrupted a phase. The
[aggregate](evaluation-panels/image-iml-zoom-v16-result.json) retains every profile,
including the failed strict audit.

All **25 authored local tests passed**, covering geometry, original source
coverage, genuine-only roles, half-even rounding, missing/tampered inputs,
component metrics and the new numerical-boundary diagnostics. The separate
[actual Linux run](https://github.com/ChipmunkRPA/veritas/actions/runs/42491297583)
passed three authored non-square geometries and15 grids, plus genuine-point
and leakage checks. Its [receipt](evaluation-panels/image-iml-zoom-v16-linux-math-review.json)
records the observed workflow commit and decoded-log digest. It tests arithmetic
within1e-12, not exact masks on real photographs or detector accuracy. Its pass
does not override the later strict real-source failure.

The [publication consistency review](evaluation-panels/image-iml-zoom-v16-publication-review.json)
also rejects six deliberate alterations of an actual metadata-only copy,
including a rewritten strict-audit status, confidence admission, changed point
seal and missing native case. The unaltered copy reproduces the aggregate.
This checks honest publication state, not detector accuracy or another model run.

From the public checkout with NumPy/SciPy/Pillow/OpenCV:

```sh
PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 python -m scripts.review_image_iml_zoom_public_v16 /tmp/new-image-crop-review.json
```

Real source/grid replay requires separately retained private inputs and permitted
cached weights. Public material contains methods, commitments and aggregates;
photos, grids, new source pairings and per-case outcomes remain private.
Outside human reviewers remain zero. Serving defaults, voice calibration and
rollback, customer model-name filtering, USD9.99 prepayment and GCP deferral are
unchanged. Regional image detection and the broader improvement program remain
unfinished; another candidate must improve useful localization and qualify
numerically stable coordinates before fresh confirmation or local integration.
