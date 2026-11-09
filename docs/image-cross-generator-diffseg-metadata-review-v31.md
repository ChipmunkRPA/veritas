# DiffSeg30k validation metadata review

DiffSeg30k remains a candidate for broader testing, but its reviewed validation
metadata does not yet support this project's source-disjoint admission rules.
No image or mask payload was downloaded, decoded, scored or used for training.
No generator or new detector weights were acquired.

## Publisher claims and observed release

The [paper](https://arxiv.org/html/2511.19111v2) describes multiple editing
generators, sequential edits and both photographed and already-generated base
images. Those categories matter: a background label for a later edit cannot
establish that the base pixels are human/camera content. This is our inference,
not an authenticity guarantee from the annotations.

The [publisher card](https://huggingface.co/datasets/Chaos2629/Diffseg30k/blob/8049819755ff430dcf4701636b14e6f1df6c4542/README.md)
declares Apache-2.0. The paper separately refers to COCO licensing. The
[official COCO terms](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm)
distinguish annotation licensing from image copyright and direct image users to
the relevant Flickr terms. This review does not establish individual-photo or
generator-output permissions from a dataset tag.

We pinned release `8049819755ff430dcf4701636b14e6f1df6c4542` and inspected only
the validation archive's directory and named JSONL sidecar. The training archive
and its metadata have **not** been reviewed. Do not generalize this finding to
every unpublished or uninspected provenance record.

## Direct findings

The validation directory has 12,004 entries: 6,001 image/mask pairs, one directory
entry and one metadata JSONL file. A separate local Python ZIP reader confirmed
the directory counts from cached metadata only; it could not access member
payloads. This was a second local implementation, not outside human review.

All 6,001 JSONL rows have exactly two fields: `image_file_name` and
`mask_file_name`, both strings. There are no source-photo IDs, photographer
identities, source-license links, base-type fields or edit histories in this
sidecar. The schema therefore does not provide the links needed to group
related sources, exclude overlap with the current study, select matched
photographed controls or separate AI-base mask semantics. Reconstruction from
other evidence remains possible; it has not been done here.

The [aggregate receipt](evaluation-panels/image-cross-generator-diffseg-metadata-v31-result.json)
contains revision, counts, metadata checksums and resource outcomes. Raw
directory listings, sidecar rows and private request receipts remain local.

## Integrity and failed-attempt record

Seven exact range requests read 1,380,934 archive bytes: ZIP end/directory
metadata and the compressed JSONL member plus its headers. The JSONL's CRC32
and declared uncompressed length matched. No PNG member was opened. Full-archive
SHA-256 was **not** verified; a publisher LFS pointer is not equivalent to that
verification.

The first directory worker exited successfully, but its monitor recorded a
missing-process sampling error and that attempt remains **failed**. It was not
relabelled or rerun. The explicit cached-directory review and subsequent
named-sidecar read each completed under a new bounded monitor with no sampling
errors or resource-limit crossings. All original logs and failures are retained.

## Decision

Do not acquire or admit this validation media into the fresh source-disjoint
benchmark yet. Inspect remaining provenance routes or use another source that
provides stable ancestry and appropriate permissions before freezing a sample.
Do not interpret a zero edit-mask value as an original-file verdict. The current
v26 model continues its already-frozen local training; its reserved calibration
and confirmation sets are unchanged. No one was contacted, no workflow ran,
and no deployment or serving-default change follows from this metadata review.
