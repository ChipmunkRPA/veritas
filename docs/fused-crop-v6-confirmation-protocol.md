# FUSED crop v6: frozen fresh confirmation

Prepared after the consumed development result and before any fresh detector
forward. The existing [development protocol](fused-crop-v6-protocol.md), thirteen
method pins and recorded gates remain unchanged. Added source binding, runner,
parity checks and tests do not retrain or alter the inference operations.

## Sources and complete denominator

The public source plan SHA-256 is
`024d8837c9ec07601f3a41efcc91d90679d9c31d3c27f9c66bcfd49e97b2b3a5`.
It selects48 unused TGIF/TGIF2 original/edit pairs: sixteen Firefly, sixteen SD2,
sixteen FLUX Fill, split equally between bounding-box and segmentation prompts.
All240 earlier source IDs and eight historical web examples are excluded.
There are96 original images, each scored natively and after JPEG75:192 planned
input outcomes and four crops per eligible input (768 scheduled crop forwards).
The formats share48 source groups, not192 independent examples.

Before freezing, rebind every source, reference mask, source archive member and
all four historical manifests. Require original/edit geometry and EXIF orientation
to match; the primary mask must equal their actual native RGB differences.
Hash every earlier original used for near-duplicate checks. Reject the entire
panel before inference if source/edit bytes reuse any historical digest, or if
the63-bit DCT perceptual distance to an earlier or within-panel original is at
most4. Do not replace cases. These checks cannot prove absence of all near
duplicates, related scenes or overlap with pretrained model data.

Publisher masks are secondary references. Verify their bytes and geometry when
available and retain unavailable entries. They do not replace the fixed primary
RGB-change mask or justify dropping an inconvenient case. All execution errors
and skipped crop checks remain in the denominator; none implies authenticity.

## Fixed method and controls

The exact development review SHA-256 is
`9c9572eb66af942b4e94f262714b9ecc89d50fbf742d57e664dc3ba6e05642cc`.
Its crop image-score gates remain `0.5474082231521608` for native and
`0.9998180270195008` for JPEG75. No fresh response selects a gate. These are
uncalibrated inspection responses, not probabilities of editing or confidence
claims. The existing commercial weights, independent runtime, MPS float32,
Pillow/JPEG codec and installed inference-package versions are bound.

Each fresh input runs the current format-specific check plus the exact same
four native RGB crops as development. No separate crop JPEG encoding, inference
augmentation, interpolation change, altered threshold, extra displayed-region
budget or confidence fusion is permitted. Existing current regions are retained;
only an assessed current result with no region can receive a crop region. The
same full-source reliability conditions, source grid and two-region cap apply.

Before fresh inference, replay twelve consumed inputs: one original/edit pair
per generator, in both formats. Choose the pair solely by minimum SHA-256 of its
source ID within the original24-group development selection. Compare every
baseline/crop array with its original committed-map receipt (absolute tolerance
2e-6), each crop score within2e-6, exact input/crop RGB commitments, exact baseline
displayed metrics and identical crop-gate and baseline-gate decisions. Small
floating differences cannot excuse a changed gate decision. Every planned case
must pass; retain a failure receipt and stop before fresh inference otherwise.

Publish and remotely verify the source/method freeze before that replay. The
fresh runner requires the independently recorded freeze SHA-256 and the hash of
the completed native parity receipt. Changed locks or failed/incomplete controls
must be refused before model loading. Preserve partial output on interruption;
never restart merely because an observation timed out.

## Review and adoption decision

Reconstruct all192 source/format outcomes, source bytes, native crop RGBs, raw
array hashes, current metrics and cropped displayed regions using the frozen
gate. Require exact planned IDs/labels and retain every failure. Report native
and JPEG75 separately, with per-generator outcomes, genuine highlights, edit
misses, mean pixel IoU/F1 and source-paired5,000-draw bootstrap intervals using
seed20261106. Misses contribute zero overlap. Reusing the source group across
formats must not double the independent sample size.

For a format to qualify for local integration, it must locate more edited
images, improve mean edited IoU, add at most one genuine-image highlight relative
to its same-input current check, and not increase unassessed cases. Require zero
crop execution failures. Preserve the failed format if only the other qualifies.
Neither a count gain nor the small-panel interval establishes population accuracy
or proves that a highlighted region was AI edited. Unknown generator/training
overlap, ordinary processing cues and changed-pixel versus semantic boundaries
remain explicit limitations.

The runner cannot enable a detector or alter overall confidence. A qualifying
format still requires source-coordinate evidence integration and UI/PDF/JSON,
model-name filtering and prepaid-analysis checks before a scoped local release.
Keep rollback. No cloud deployment or outside human review is established here.
