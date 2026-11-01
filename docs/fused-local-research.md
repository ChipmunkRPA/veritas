# FUSED local compatibility research

Status (2026-10-31): a separate, independently implemented application runtime is
registered **disabled by default** and enabled only in the ignored local `.env`
for limited inspection. The research evaluator and downloaded reference source
are not imported by the application. The work below establishes bounded
implementation and evidence-chain checks, not accuracy, proof of editing,
calibration, or service admission. Earlier results remain preserved as dated
records; the later holdout and writer-precision amendment do not replace them.

## Provenance and licensing observations

- Architecture source inspected at [AntonNuzhdin/FUSED, revision
  e8580ebad33decf0787a96133af4fafee640e024](https://github.com/AntonNuzhdin/FUSED/tree/e8580ebad33decf0787a96133af4fafee640e024).
  Its complete Git tree returned no path containing `licen` on 2026-10-31. Source
  redistribution/commercial use permission has therefore not been established.
  The independent evaluator does not vendor or import that source. Five reviewed
  reference files were subsequently downloaded to ignored local `data/` storage
  for the isolated numerical parity check described below; they are not shipped
  with the application.
- The [FUSED checkpoint card at revision
  b7dfe0cb2d2f8fc8fd9e6ebea5f7185cfa8e1af7](https://huggingface.co/aonuzhdin/FUSED/blob/b7dfe0cb2d2f8fc8fd9e6ebea5f7185cfa8e1af7/README.md)
  declares MIT. That declaration is recorded for the weights; it is not treated as
  automatic clearance for the separate source repository.
- The [timm ConvNeXt XXL checkpoint card](https://huggingface.co/timm/convnext_xxlarge.clip_laion2b_soup)
  declares Apache-2.0. The evaluator uses the installed `timm` implementation and
  the downloaded safetensors, with network model loading disabled.
- Test data remain in ignored local storage. See the pinned dataset manifest for
  original image/mask hashes, source revision, selection rule, and attribution.

## Safe loading and implementation boundary

The standalone `backend/scripts/fused_research_eval.py` is a functional tensor
interpreter for this specific checkpoint. It has no training implementation and
does not support arbitrary checkpoint/config inputs. It checks both SHA-256s:

| File | SHA-256 |
| --- | --- |
| fused_opensdi.pth | 7e527d49080a773641cf96915b4b3fa12d571c5b87f35c527f13ab82f148364c |
| convnext.safetensors | 1ddf6dbc7f0753fc4dfec5db5fa474acfd1ac6f67f466d6a976780c30b3a1b23 |

The PyTorch checkpoint has no reported unsafe globals. It is loaded only with
`weights_only=True, mmap=True`. The frozen backbone uses safetensors. Its module
is constructed on the meta device without downloading or initializing random
weights, then strictly assigned the complete nonclassifier state. Missing tensor
keys in the functional path fail immediately. Unused checkpoint keys are recorded.

The published checkpoint contains legacy `dinov2_*` buffers and `seg_bridge.*`
parameters that the inspected current inference path does not use. It also saves
unused SparseViT normalization/positional parameters and training counters. These
must not be mistaken for a successful independent numerical parity check.

Three small CPU tests verify strided attention coordinate grouping/inversion,
pooling equivalence with PyTorch's MHA module, and expert routing equivalence with
explicit per-token weighted experts. The bounded reference parity run below
additionally checks the complete native-resolution inference path on ten inputs.

## Local protocol

Hardware reports 18 GiB physical memory and Metal availability outside the tool
sandbox, with a recommended GPU allocation of approximately 13.3 GiB. The frozen
backbone contains 843,398,016 parameters; its file is about 3.2 GiB, plus a roughly
305 MiB trainable checkpoint. Run one inference process at a time while the local
API's large inference models are released.

```sh
backend/.venv/bin/python backend/scripts/fused_research_eval.py \
  --device mps --per-group 1 --output fused-research-f32
```

The selection is the first stored example in every family/label group of the
existing 200-sample manifest: five generator families, one edited and one authentic
image each. No score-based selection or threshold tuning. Stored image coordinates
are preserved; neither image nor mask receives an independent EXIF transform.
Images are converted to RGB and PIL bilinear-resized to 512 square. The forensic
input uses ImageNet statistics and the semantic input uses saved CLIP statistics.
Mask probabilities are bilinear projected to the original stored dimensions and
thresholded at 0.5. Compared with the upstream example, this retains float32 mask
values instead of first quantizing to 8-bit before resizing; that protocol
difference is intentional and recorded, not claimed bit-equivalent.

Outputs are ignored local files under
`backend/data/localization-eval/fused-research-f32/`: per-image float probabilities,
binary masks, and incremental `results.json`. This is a convenience smoke sample,
not a representative product validation or a proof of where editing happened.

## Original convenience-sample result (2026-10-31)

Both the 10-image smoke test and a subsequent fixed 200-image run completed in
float32 on MPS. The second run uses every existing manifest item (20 edited and
20 authentic images per family); no further data were downloaded. It loaded in
3.63 seconds and spent 118.81 seconds inside image inference/projection, with a
median of 0.585 seconds per image. End-of-run allocations were 3,693,177,088 bytes
for tensors and 4,741,545,984 bytes reported by the MPS driver. These are sampled
end-of-run values, **not measured process or GPU peaks**.

Results are in `backend/data/localization-eval/fused-research-200-f32/results.json`.
Its sibling `region-audit.json` records a separate offline audit. That audit uses
fixed 8-connected components of at least 0.001 of the original image area, mask
threshold 0.5, and separately reports an image-classifier gate of 0.5. These
policies were specified before this audit; no threshold was fitted to these data.
`run-provenance.json` binds the executed evaluator/audit/test file hashes, result
hashes, and installed library versions. The first fixed SD1.5 edited/authentic
pair is visualized in `fused-research-f32/first-pair-diagnostics.png`; this is a
diagnostic example, not a score-selected performance summary.

| Family | Edited / authentic | Edited mean IoU after component filter | Edited mean pixel F1 | Authentic images with retained regions |
| --- | ---: | ---: | ---: | ---: |
| SD1.5 | 20 / 20 | 0.521 | 0.657 | 20 / 20 |
| SD2 | 20 / 20 | 0.509 | 0.629 | 20 / 20 |
| SDXL | 20 / 20 | 0.395 | 0.500 | 20 / 20 |
| SD3 | 20 / 20 | 0.544 | 0.664 | 20 / 20 |
| FLUX | 20 / 20 | 0.216 | 0.310 | 20 / 20 |
| All | 100 / 100 | 0.437 | 0.552 | 100 / 100 |

The raw localizer therefore shows substantial false regions, not merely isolated
single pixels: authentic images have 4.11% mean flagged area after the component
filter. Gating the same masks by the image classifier reduces authentic images
with regions to 6/100 and mean flagged area to 0.20%, while 7/100 edited images
are classified below the gate; edited mean IoU falls to 0.422 and F1 to 0.530.
The classifier has 93/100 edited positives and 6/100 authentic false positives
at the fixed gate. Neither policy is selected or admitted to the product.

This establishes local resource feasibility and identifies a serious localization
limitation in this compatibility experiment. It does not establish published
model accuracy or independent parity with the authors' complete runner. The
script read 815 of 879 checkpoint tensors during inference; all 64 unused keys
are explicitly recorded and match saved parameters/buffers unused by the
inspected current forward path. The exact source-to-checkpoint compatibility
was subsequently checked on a bounded ten-image reference run below.

## Reference parity (2026-10-31)

`backend/scripts/fused_reference_parity.py` compares the independent functional
evaluator with the five hash-pinned reference source files in ignored local
storage. It does not import the upstream evaluation CLI, deserialize arbitrary
pickles, or enable remote weight loading. Both paths use the exact same pinned
weight tensors and installed `timm` backbone; architecture computation is separate.
The original constructor's pretrained-backbone request is replaced by that
already loaded local backbone. Missing required state or unresolved meta tensors
fail the run. The reference code is not an application dependency.

Selection was fixed before running: first image in each family/label group of
the existing manifest (ten inputs). Both paths use the same RGB/bilinear-resized
512-pixel input and independent normalization expressions. Native 512 mask
probabilities and image responses are compared before any source-size projection,
with predeclared absolute/relative tolerances of 0.0001. On float32 MPS, all ten
passed. Maximum mask absolute error was 0.00002462; maximum classifier absolute
error was 0.0000008941. One pixel in one image crossed the fixed 0.5 mask boundary;
this is numerical agreement within tolerance, not bit-identical masks.

`backend/data/localization-eval/reference-parity.json` retains all source/input
hashes and per-image errors. This resolves the bounded implementation-parity
check only. It does not prove general accuracy or equivalence for other devices,
precision modes, weights, input sizes, or projection policies.

At this stage, model integration, a disjoint holdout and operating-policy
evaluation remained follow-up requirements. The subsequent work below addresses
those bounded checks while retaining separation from the unlicensed reference
source. It does not turn these masks into proof of AI edits or localized forgery.

## Disjoint fixed holdout and recorded-grid audit (2026-10-31)

The next 200 inputs are under
`backend/data/localization-eval/fused-holdout-offset20-count20/`. The manifest pins
`nebula/OpenSDI_test` revision `7e233eaf98fcfee4c74c788f0e34d06feb7ad0df`.
Selection scans physical row order, skips the first 20 unique partial-image
bytes per family/label, excludes every original manifest hash, then takes the
next 20. There are 100 edited and 100 authentic images across the same five
families. None share image bytes with the earlier 200-image manifest
(`0deae28e2e5fd0fcf9c839583c5a36a1e8a4ec8282a87a8d0a2f84a4c5e3ee84`).
Byte disjointness does not establish unrelated source scenes, independence from
training data, representative sampling, or a general product error rate.

Image gate and mask threshold were both fixed at `>= 0.5` before this inference;
neither was fitted to the holdout. The recorded-grid audit projects the native
512 map to the aspect-correct screening grid (long side at most 1024) using
OpenCV bilinear interpolation. It applies the existing deterministic quality
mask, 8-connected components of at least 0.001 of grid area, the ten-region
runtime cap, then the two-region presentation cap. Source projection uses the
same Pillow nearest-neighbour geometry as evidence overlays and preserves holes
inside components. Metrics cover the entire reference image: excluded quality
regions still count as possible misses. All 200 images pass the recorded
EXIF/minimum-size comparison conditions.

| Fixed holdout policy | Edited mean IoU | Edited mean pixel F1 | Edited images with zero overlap | Authentic images with retained pixels |
| --- | ---: | ---: | ---: | ---: |
| Ungated source-size raw threshold | 0.453910 | 0.558618 | 5 / 100 | 100 / 100 |
| Ungated recorded-grid, quality/component filters, two-region cap | 0.362627 | 0.470005 | 8 / 100 | 100 / 100 |
| Same recorded-grid policy plus fixed image gate | 0.354384 | 0.456954 | 13 / 100 | 4 / 100 |

The fixed image gate passes 93/100 edited images and 4/100 authentic images.
After quality filtering and the presentation cap, 13 edited images have no
retained overlap. These results support an explicitly limited inspection aid
only. A failed gate withholds candidates while preserving raw measurements; it
must not be described as authenticity. A passing gate does not validate any
pixel as edited. The audit covers one map and its component caps, not competition
between different checks, the total packet cap, or a native-tile mode.

## Writer-precision amendment (2026-10-31)

A real `ExhibitWriter` integration test found that the old adapter and audit
rounded `float32 * 255`, while the persisted PNG writer first promotes to
float64. For example, the actual value of float32 `0.9` encodes as 229 in the
writer, while multiplication in float32 reaches a half tie and encodes as 230.
The application adapters now use float64 quantization and float64 decoded values
for component statistics, matching the writer and evidence reader. The model's
unquantized output is unchanged.

`backend/scripts/audit_fused_writer_quantization.py` replays every saved holdout
prediction on CPU without new inference, data selection or threshold changes.
It writes a new `writer-precision-amendment/` policy, results and scalar,
reliability, and capped-mask PNGs. Prior policies, predictions and audit files
remain unchanged. It verifies source image/reference hashes, native and
source-size predictions, prior scalar/quality/cap masks, and original metrics.

The corrected precision changes **250 encoded cells across 117 images**. On this
holdout it changes no threshold memberships, component counts/geometries,
rounded component statistics, capped selections or final metrics. A synthetic
regression explicitly shows that such rounding **can** change component rank
and which two regions survive a cap; the correction cannot be dismissed simply
because this holdout's selected regions happen to remain unchanged.

## Application runtime and local evidence verification (2026-10-31)

`backend/app/detectors/image/fused.py` carries our independent tensor operations,
not imports from either research script or the separately downloaded reference
source. The dedicated `adapters/fused_loader.py` verifies both pinned local files
before deserialization, refuses unsafe globals, uses `weights_only=True` and
`mmap=True`, checks the exact static configuration and 879-tensor schema, and
strictly assigns the safetensors backbone built on the meta device with
`pretrained=False`. It supports one already-oriented RGB image, a 512-square
screening resize and float32 CPU/MPS; it has no automatic downloads or tiles.

The fixed first edited and first authentic holdout records were run through the
application runtime on MPS. Both native float32 masks are bit-identical to the
saved independent evaluator arrays, including their NPY SHA-256s; both scores
are bit-identical, and zero pixels cross 0.5. This is a two-input runtime
regression, separate from the ten-input reference-parity test and from accuracy.
Load time was 5.74 seconds; the two inferences took 1.11 and 0.60 seconds. Resident
tensors occupied 3,693,174,404 bytes. Ten-millisecond sampling observed maximum
MPS tensor allocation of 3,837,094,656 bytes and driver allocation of
4,610,473,984 bytes; these are sampled maxima, not true peaks. MPS tensor
allocation returned to zero after release.

The managed composite's declared weights floor is 3584 MiB and its one-input
activation allowance is 2048 MiB, both estimates. Its dependency is verified and
loaded inside the composite rather than leased a second time. Registry entry
`loc.fused@0.1.0` remains default-disabled, uncalibrated, at zero legacy/fitted
decision weight, and pending admission. Only the ignored development `.env`
allowlist enables it for the local demonstration. This is not approval for a
cloud deployment or general service operation.

The same two source JPEGs were uploaded unchanged with neutral filenames in two
new full-profile analyses, recorded in
`backend/data/image-evidence-v3-demo/analyses.json`:

- `editing-example.jpg`: `fbee815b-3801-4d44-8781-7b6353600a07`; exact image
  response `0.9999998807907104`; one raw/eligible/displayed component.
- `original-photo.jpg`: `aa5cd758-b4b1-4d65-b025-d8c1ab762ba0`; exact response
  `0.14113351702690125`; three raw components, zero eligible/displayed components.

The initial control `8b037bb0-1f5d-4ae1-8f53-22f4be4ae3a7` is retained unchanged.
The replacement is a new analysis of the same bytes after correcting the
no-selected-region explanation; it is not an edit to a previous result.

The analysis scalar and quality PNG byte streams exactly match the amended
frozen audit, and the selected masks match exact connected-component membership.
The failed gate creates no relative-threshold fallback localisation. Public
evidence equals the JSON export; 35 unique asset hashes and nine source crops
were checked. The pre-job baseline confirms 23 earlier analysis results and 17
completed report PDFs/snapshots remain unchanged. These are engineering fixtures,
not examples selected to characterize detection accuracy. Their separate
`verification.json` records the completed public-copy checks: analysis details,
evidence and JSON exports contain no internal model identities. The initial
generated-copy identity leak and prior demo list are separately preserved for
the audit. PDF/package verification is recorded separately by the integration task.

Thirty-five focused CPU tests cover safe loading/configuration, tensor
operations, gate/raw-measurement separation, single-image full/deep adapter
execution, persisted source/scalar bindings, invalid output refusal, and the
writer-precision/ranking regression. No large checkpoint is loaded by these
default tests.

## Frozen artifact commitments

The following artifacts live under ignored
`backend/data/localization-eval/`. The `holdout/` abbreviation in this table means
`fused-holdout-offset20-count20/`; original and amended artifacts are both kept.

| Artifact | SHA-256 |
| --- | --- |
| `reference-parity.json` | `70de3cd897748715fae7c87516982c30d6628ba9f930b3d3e3acb94bbb6789c1` |
| `holdout/manifest.json` | `9c90cc3f9bbe0f2dfd29e7cc36eb06066ebf0800d9a0281aa0ad6224b08be036` |
| `holdout/evaluation-policy.json` | `75a5bb3716386dfbdadac1d5600c7b72f81916c3ecdb9ff0f0d03b91a548e912` |
| `holdout/production-grid-policy.json` | `61e36282ce005fc68f73afe67c119548a9d3434a439a0582cd52a0f6ca98bd18` |
| `holdout/fused-research-f32/results.json` | `e4f51e95830204776800be498ff6f066ec17520efe55be35e7525e63656132a2` |
| `holdout/fused-research-f32/production-grid-audit.json` | `fab6e48d085cb1adf16dfe3112bd3b3dbc38bdb132259660833d2c03ad3e7c00` |
| `holdout/application-parity/results.json` | `e512bd6da468e49fa5ba7cbe06702cce4fbaa344288370d4e96aebb0c2d166ca` |
| `holdout/writer-precision-amendment/amendment-policy.json` | `f1329fb07db68f013d3856c67eb1fe282006a5e566f902b28b2f898e7ae294d2` |
| `holdout/writer-precision-amendment/amendment-results.json` | `59e4d35b51da9e6ebfe75945bacaf661c415b9a27cf0bd2e192aabf7325ba9b5` |
