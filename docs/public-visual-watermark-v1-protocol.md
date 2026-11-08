# Public visual watermark authored contract v1

Freeze code, asset identities, readout and cases publicly before model
construction or inference. This is a real execution of a public spatial
watermark model on non-neural authored RGB carriers. It is not real-media
accuracy, provider verification or passive deepfake detection. A compatible
public mark can be added to genuine content.

Use the explicitly **MIT SA-1B WAM** object, never the noncommercial original
COCO release. Pin source2c08af04d037d5667c02f6ddebbda9ff04581c3e and publisher
mirror88ffb2cd1079a76b01d6beda8bf2a433c25fa881. The explicit MIT S3 object must
match SHA25690ef232384e023bd63245eb0c131abd69d2afc7b8f17a71ccedceb542bf009e2,
377825938bytes. Keep publisher MIT, adapted component notices and isolated
OmegaConf/ANTLR source identities. Do not install packages or change serving
dependencies. Source inference imports were checked without constructing models.

Local disk cannot accommodate larger retained checkpoints while preserving the
1536MiB reserve. Read this pinned checkpoint directly into bounded RAM, verify
its full byte count/SHA256 before tensor loading, require no extra pickle
globals, and use Torch weights_only=True on CPU. Validate finitefloat32 tensors,
strictly load the reference architecture and compare every final state tensor.
No upstream unrestricted loader, hub or unplanned network is permitted. Do not
save source checkpoint bytes on disk; retain their identity and loaded-state
receipt. Exact neural replay requires reacquiring that pinned object.

Reference: [source/weights licensing](https://github.com/facebookresearch/watermark-anything),
[model publisher](https://huggingface.co/facebook/watermark-anything).
The original inference utility uses unrestricted torch.load; it is inspected,
not called. Use its configs and the native embed/detect architecture with the
original JND settings, scaling_w2, ImageNet preprocessing and fixed alternating
32bit message. Six256x256 authored sources: gradient, checker, waves, seeded
texture, flat gray and rings. These are not human-created photos or provider
outputs and cannot establish population accuracy.

## Images

Six carriers × four states × three transforms =72 cases. States: no added mark;
full native mark; partial center128x128 mark at x64,y64; ordinary rotated center
crop with no mark. Generate a full native mark, unnormalize/clamp and quantize
RGB8 by floor(value*255+0.5). Partial copies only the marked center pixels;
every other source pixel must stay bit-identical before transforms. Ordinary
editing alone must not be treated as AI truth.

Transforms: native RGB, JPEGquality70, and bilinear down128/up256. Save original,
marked/partial/ordinary source arrays and every exact detector input. Intended
region coordinates remain in the256x256 source grid; resampling/codecs can spread
signal and do not establish exact edited-pixel boundaries.

## Frame sequences and codecs

Three authored moving8frame clips at2fps (gradient/waves/texture). Unmarked and
partial states, each tested as raw RGB and H264CRF23 =96 frame cases. For the
partial state, generate marks on frames2:6 only; copy only the center128x128
pixels. The intended mark occupies source1–3seconds. Before, during and after
frames stay identifiable by exact frame index, source PTS and rational timebase.
Decode all8 frames without trimming, padding or frame-rate conversion and require
PTS*timebase=index/2. Retain all six MP4s and exact input RGB arrays.

## Fixed readout, counts and independent replay

Use first-channel native logits strictly>0. Preserve every presence logit and
pixel. Eight-connected components with at least500 pixels are proposed inspection
regions, not a calibrated false-alarm or AI-confidence rule. Keep raw selected
pixel counts even below500. Record boxes, source-grid masks and overlap against
the intended marked region; do not claim these are general AI-edit masks.

Use the publisher-supported hard message method: each bit logit>0, then majority
over the selected mask pixels. Preserve all32 binary maps packed losslessly.
Raw continuous bit logits are unused and not retained. An empty selected mask
has unavailable message/bit accuracy. The supported hard helper must agree with
the replayed message. Do not infer a provider or creator from those bits.

Expected:18 generator forwards (six images plus12 marked video frames),168
case detector forwards, four wrapper/direct parity comparisons and one bit-exact
partial repeat =173 detector forwards,191 total. CPUfloat32, twoTorch threads,
seed20261108. Retain every failure, false response and miss; no point adjustment
or post-result source/variant removal.

Run a separate model-free replay of all168 case identities, RGB bytes and source
transforms, partial spatial/time boundaries, every retained presence pixel and
binary bit, independent component boxes/majority/overlap arithmetic and all48
codec frame re-decodes/source times. This is code/arithmetic review, not outside
human validation. Code, methods, hashes and aggregates may be public; authored
arrays/private journals remain local.

Bound native and review jobs to900seconds,6GiB sampled process-tree RSS and
1536MiB available RAM/free disk; report guards/sampling errors. Public-family
presence remains outside serving, overall confidence, provider authentication
and the trusted AI-declaration early stop. No customer input, payment, provider
upload, GCP action or human outreach occurs. PixelSeal/VideoSeal checkpoints and
private Muse compatibility are separate, unqualified work.

A freeze-only preflight stopped on an OpenCV distribution-name mismatch before
any model construction/inference. Its record is retained; the inventory now
names the installed headless distribution. Cases/readout are unchanged.
