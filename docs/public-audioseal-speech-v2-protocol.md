# Public speech watermark and background qualification v2

Freeze this separate study before neural model construction or inference. Keep
the completed v1 authored trial and its outcomes unchanged. This study tests
public-family watermark presence and time localization, **not AI authorship,
private provider compatibility or voice deepfake classification**. No public
screening threshold or confidence guarantee is introduced.

## Public assets and reproducibility

`acquire_public_audioseal_v2.py` downloads or copies exactly pinned publisher
assets from `public_audioseal_assets_v2.json`: source revision
1496f7662d6aa625602bb0dcd1b4997b63b47609, model revision
3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba, two weights, two dependency archives,
the pinned model card and ANTLR license. Verify actual sizes and SHA256, not
HTTP ETags. Extract only 35 reviewed source and 80 dependency files with their
individual identities; retain license notices. Do not install packages, invoke
setup scripts, resolve checkpoint training configuration or run upstream
unrestricted checkpoint/network loaders. Frozen v1 restricted loading must
verify every final tensor. Source and weights are MIT; dependency notices are
retained. Publisher references: [AudioSeal](https://github.com/facebookresearch/audioseal),
[model release](https://huggingface.co/facebook/audioseal).

## Source plan and stopped preparation

Use exactly 32 known-unused LibriSpeech speakers from the first eight pinned
train.clean.360 shards, selected by salted identity order before inference.
Known canonical speaker IDs from prior project source plans/manifests, including
background speakers, are excluded. Select a hash-ordered utterance of at least
eight seconds, then its first 128,000 mono16kHz signed16 samples. External
pretraining overlap is **unknown**. English read audiobooks are one domain;
this study does not cover conversations, languages generally or new providers.
LibriSpeech is [CC BY4.0](https://www.openslr.org/12/); retain source attribution.

The first source preparation stopped on a background PCM checksum mismatch,
before model construction/inference. Its files/log/profile/code are retained.
The first check hashed signed16 bytes against a saved checksum of the
production float32 representation. A direct comparison confirmed equal sample
values for the first flagged source; this was a checksum-representation error.
Recovery preserves the 32 already collected original foregrounds
and exact inputs, verifies them again, and uses the production decoder for
backgrounds, requiring the saved canonical PCM checksum. Earlier rejected short
candidate bytes were not retained; the recovery record discloses that limit.
A separate recovery attempt could not access process monitoring, stopped before
source work, and is retained. The successful recovery records its resource
sampling error count rather than suppressing it. No failed native inference is
being rerun or excluded.

Backgrounds come from the existing identity-audited plan: project-consumed
public-domain MUSAN noise, CC BY3.0 music and CC BY4.0 speech, with exact source
checksums, license/attribution, chosen offsets and canonical speaker identities.
They are **reused backgrounds**, not fresh independent nuisance samples. Choose
one hash-selected eligible excerpt per kind and foreground. Known speaker
overlap with the foreground must be zero. Preserve the recovered plan before
inference; no detector response controls any source selection.

## Fixed 576 cases and inference

32 foregrounds × three mark states × six conditions. Foreground RMS0.1 in
float32; fixed alternating16bit message; frozen native generator on the full
eight seconds. The partial mark copies only source samples32000:96000
(2–6seconds), leaving all other unmodified source samples bit-identical.

Conditions: clean; white noise0dB; environmental noise0dB; music0dB; other
speech0dB; and64kbps MP3 of the clean gained input. Background realization and
gain use the unmarked foreground RMS and are identical across all three mark
states. One float32 clipping-safe gain per speaker, min(1,0.8/pre-codec peak),
applies to all conditions and states. Every exact model input is retained.
MP3 is decoded to float32 mono16kHz, requiring exact128000 samples without
hidden alignment trimming or padding. Preserve the96 codec files.

Use the unchanged v1 native readout: sample response strictly>0.5; a250ms bin
selected only when more than half its4000 samples exceed0.5. Keep all samples,
bins and joined source-time intervals. The response fraction is not AI
confidence; per-sample output does not establish sample-accurate boundaries.
Expected forwards:32 generator,576 direct detector,4 convenience parity and
1 bit-exact repeat =613. CPUfloat32, two Torch threads and fixed seed20261108.
Retain every case and failure. No point adjustment, source replacement or
post-result variant removal is allowed.

## Results and review

For each condition separately, show unmarked-control selected cases/32, full
and partial misses/32, interval IoU, marked-time coverage and time selected
outside the known mark. Give Wilson95 intervals for file rates. The same32
speakers recur across conditions;576 cases are not576 independent sources.
A deliberate mark on genuine speech does not establish AI generation.

Run a separate model-free review of all source/mixture/input hashes, matched
background and partial geometry, native samples/bins/intervals, independent
interval-overlap arithmetic and96 exact codec re-decodes. Preserve original
and independently recomputed aggregate denominators. This is code/arithmetic
review, **not outside human validation**. Keep raw speech, per-case pairings and
sample responses private; publish methods, code, hashes and aggregates.

Bound each native/review job to900seconds,6GiB sampled process-tree RSS and
1536MiB available RAM/free disk. Report sampling errors and guards. No serving
admission, overall-confidence contribution, provider authentication or trusted
AI-declaration early stop follows from this study. No customer inputs, payments,
provider upload, GCP action or human outreach occurs.
