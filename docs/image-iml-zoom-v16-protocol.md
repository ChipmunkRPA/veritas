# Magnified overlapping image crops: prospective development protocol

Test whether the existing permitted IML-ViT localizer finds more small edits
when it inspects magnified crops. Earlier tiling used a different residual
localizer and failed; it does not test this transformer's receptive-field
context. No checkpoint training, parameter search or post-result gate change.
Freeze and publish code, input/weight identities and this protocol before any
new project-model response. This is consumed-source development, not fresh
confirmation or improved customer confidence.

## Exact roles and controls

Use the exact 24 source IDs in the old IML development policy from the retained
TGIF-v2 panel as genuine point controls, with no AI images in that role. Use all
48 source pairs in the old v4 confirmation panel as consumed diagnostics. Roles
must be source-disjoint. Both retain Firefly, SD2 and FLUX Fill strata. No source
is replaced because of acquisition, geometry, prediction or failure.

Each point original supplies itself and an ordinary edit, in native PNG and
Pillow JPEG75: 96 file cases total. Each diagnostic source supplies original,
AI edit and ordinary edit in both formats: 288 file cases. Source variants are
correlated; these are 72 source groups, not 384 independent photographs.
The ordinary operation blurs the original with Gaussian radius2 and brightens
it by1.08 only in the central quarter of image area, using a16-pixel linear
feather. It adds no AI generation. JPEG follows the operation and does not add
AI truth. Edited reference masks remain the exact original source-pair mask,
not every pixel affected by JPEG. Ordinary/original truth stays entirely zero.

## Fixed multiscale map

Keep the existing pinned, safely loaded MIT IML checkpoint and independent
compatible head. The owner explicitly permits commercial checkpoint use in
[issue60](https://github.com/SunnyHaze/IML-ViT/issues/60); code/checkpoint/data
permissions remain separate. No restricted decoder is imported and no weight
is downloaded. The existing encoder/FPN, normalization and top-left padding
remain unchanged. All212 model tensors load strictly.

Four corner crops each cover ceil(.6*width) by ceil(.6*height). Magnify each to
long side1024 with Pillow bilinear resize, preserving aspect ratio to nearest
integer and retaining its source rectangle. Do not re-encode a cropped JPEG.
The whole RGB input is decoded once; original JPEG traces predate cropping.
Each actual normalized forward input must match the expected source/crop tensor.

To keep complete replay within available local storage, the scientific readout
is explicitly a lossless256x256 uint8 grid per whole/crop response. Reduce the
native float32 response with canonical float64 half-pixel bilinear sampling,
then round-half-even to255 levels. Check every reduced floating value inline
against a separately implemented bilinear reference within1e-12. This is a
predeclared readout change, not a claim to retain every native model output.
All five recorded grids remain available privately for complete reconstruction.

Project grids back to their exact native rectangles with the same sampling.
Exclude six percent of each tile's internal borders; preserve source-image
edges. Average all valid tile maps at each source pixel and require full source
support. The primary map is half that average plus half the whole-image map.
The matched control uses the same recorded whole-image grid without tiles.
Thus coarse recording is shared by both arms. Preserve reliability exclusions,
fixed pixel cutoff.5, component rank/filter and two-region per-map display cap.
These are inspection regions, not pixel-authorship probabilities.

The first clean case in each native phase repeats its full prediction and first
tile exactly. Its whole result also agrees with the retained same-input full
map within1e-4 and has identical recorded grid bytes. Other old whole maps are
reused under input/map commitments, not re-inferred. Ordinary edits require a
new full forward. Every new actual input is hash-bound; no missing model/input
can fall back to initialized weights, zeros or an originality verdict.

## Genuine-only points and comparison

For each format/arm, use all24 genuine source groups, taking the maximum active
source-pixel fraction over original/ordinary where an ungated region exists.
Active means the recorded source display value reaches.5. Choose
max(.001,nextafter(second-largest genuine-group value,+infinity)), allowing at
most one of24 groups to display regions. Exact ties remain explicit. Save both
format/arm points and the completed journals before any diagnostic processing.
No AI response or diagnostic label chooses a point.

Compare the primary with the matched whole control and the actual currently
selected native/JPEG component on the same diagnostic sources. The current
original/AI responses are reused with complete source/map/postprocessing replay;
all96 ordinary-reference cases are newly inferred. Repeat the first reference
input for each format. Current model/loader/component hashes and weights must
match the retained reference contract. No raw checkpoint score is confidence.

Report each format, generator, operation and all planned denominators: any
overlap, useful IoU>=.3 locations, mean IoU and pixel F1 with misses as zero,
genuine/ordinary highlights, assessed/error counts and paired source bootstrap
differences (5,000 draws, seed20261108). No variant counts as a new source.
Each predeclared format qualifies further confirmation only if the primary
beats both references on useful locations and mean IoU, loses no current useful
location, adds at most one original and one ordinary highlight versus current,
does not exceed the matched control's genuine/ordinary highlights, and loses no
coverage. A failed format stays failed; no successful generator is substituted.

## Resource, source and independent checks

Run point-native, point-JPEG, seal, six format/family diagnostic phases, two
ordinary-reference phases, summary and separate audit sequentially. Each heavy
phase has a900-second/6-GiB sampled-RSS guard and requires1536MiB available RAM
and disk. Preserve sampling errors and failed/partial attempts. Never silently
restart, overwrite a journal or substitute sources. Raw model grids and current
reference arrays remain private; no new image download or checkpoint cache.

The separate audit reconstructs ordinary/JPEG inputs, crop coordinates,
recorded-grid interpolation/stitching, reliability/components, genuine-only
points, every source-mask intersection and all counts/gates. It repeats no
neural forward and inherits the inline native-output reduction/input checks.
This is not a second execution of every new model response. Authored tests cover
non-square/odd geometry, all four edges, exact source coverage, interpolation,
flat/tied controls, source-role isolation and point/packet tampering.

TGIF/TGIF2 publisher metadata retains CC-BY-SA4.0; original photo rights remain
separate. No photos, new private pairings, native maps or customer uploads are
published. Share methods, commitments and aggregates only. Outside human review
remains outstanding. A development pass still needs unused-source confirmation,
independent evaluation and full UI/PDF/JSON/prepaid-flow tests before a scoped
replacement. Serving, public confidence, model-name filtering, Stripe USD9.99,
voice rollback and GCP resources remain unchanged.
