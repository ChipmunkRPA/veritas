# Image training-data audit: a smaller, source-grouped review pool

**Do not fine-tune on the full image benchmark by assuming its dataset license
clears every photograph.** The pinned publisher metadata contains2,242 distinct
COCO photo IDs. Only597 have the recorded BY/BY-SA2.0 schemas accepted by this
preliminary commercial/adaptation filter. Photographer attribution and the
link to the actual higher-resolution source pixels remain unverified; **no
photo is admitted to production training by this audit**.

This is the next methodological step after the
[rejected full-source localizer](image-relayformer-v18-result.md). Its raw maps
often highlight broad original/ordinary regions while missing actual AI edits.
Targeted regional learning needs suitable positive regions and ordinary-edit
controls, with source-disjoint evaluation. Another point adjustment on the same
consumed images does not solve that problem.

## What the local inventory establishes

The audit downloads no new photograph, runs no neural model, fits nothing and
changes no serving default or confidence calculation. It reads15 pinned TGIF
metadata CSVs, the pinned publisher license/README/preparation source, official
COCO terms/download text, and both caption-image metadata members of the official
2017 annotation archive. All inputs are hashed; acquired Python is saved as text
and never executed.

| Source group | Distinct photos | Recorded BY | Recorded BY-SA | Recorded NC | Recorded ND | Other review | Preliminary compatible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Publisher training | 1,558 | 292 | 140 | 1,035 | 90 | 1 | 432 |
| Publisher validation | 341 | 54 | 28 | 233 | 24 | 2 | 82 |
| Publisher testing | 343 | 54 | 29 | 250 | 10 | 0 | 83 |
| All distinct photos | 2,242 | 400 | 197 | 1,518 | 124 | 3 | 597 |
| Already used in our local comparisons | 240 | 38 | 20 | 175 | 7 | 0 | 58 |

NC and ND are screened out under this project's conservative training rule.
Unknown/conflicting URLs and other labels also require review. The exact license
URL and photo record determine the classification; a dataset title or annotation
license cannot substitute for them. BY/BY-SA compatibility is a metadata finding,
not final permission, legal advice, a privacy release or camera-origin proof.
This result does not retroactively establish or adjudicate the rights of every
previous evaluation use.

## Source grouping and the misleading FLUX original files

The2,440 training rows are **2,440 distinct source/category pairs but1,558
distinct photographs**, with up to seven categories for one photo. They are not
2,440 independent source groups. All category, generator, mask, compression and
ordinary-edit variants must remain in one group when fitting or evaluating.
There are no training/validation/testing photo-ID intersections in the base
original CSVs or the actual Firefly,SD2,FLUX Fill generation CSVs at this revision.

However, `orig-flux_training.csv`, `orig-flux_validation.csv` and
`orig-flux_testing.csv` are byte-identical and all list the same331 test-source
photos. This is a specific metadata inconsistency, **not evidence that the
actual generation splits overlap**. Our review queue uses base original IDs and
actual generation IDs; it never assigns roles from those three FLUX original
filenames. FLUX Fill lacks45 base training,11 validation and12 testing sources;
those cases are explicit and are not silently replaced.

After the preliminary license filter, requiring records for all three generators,
and excluding every locally consumed source, the private rights-review queue has:

| Publisher role | Compatible across all three generators | Already consumed locally | Queued for rights/lineage review |
| --- | ---: | ---: | ---: |
| Training | 420 | 0 | 420 |
| Validation | 79 | 0 | 79 |
| Testing | 78 | 54 | 24 |

The523 queued photos are **not admitted training data or frozen experimental
assignments**. Not locally consumed does not establish freshness or absence of
external model-pretraining overlap. The24 remaining test sources are also too
few for the earlier48-source confirmation design; do not shrink that requirement
or reuse consumed photos under new names. A strong confirmation needs additional,
separately permitted sources.

## Why the pixel lineage still matters

The publisher's inspected preparation code takes the COCO Flickr URL, requests
the larger image size, may rotate an image based on aspect ratio, and can use a
backup original when retrieval fails. Its model-specific crops differ between
the base and FLUX archives. A matching COCO number or the smaller COCO JPEG
therefore does not establish the exact pixel lineage of a TGIF original.

The audit verifies all existing local original-image hashes and all240 consumed
source groups, but **does not claim** it reconstructed every original from its
Flickr source. All240 currently have one retained original digest in these four
manifests. Photographer attribution, larger-source bytes, orientation and crop
reconstruction are the next concrete checks. Source terms, dataset ShareAlike,
generator-output terms, privacy and any additional rights remain separate.

## Verification and publication

All52 local authored tests pass without skips. They cover NC/ND/unknown metadata,
annotation-license substitution, misleading URLs, duplicate photo grouping,
split conflicts, missing generator sources, used-test exclusion and refusal to
turn a review queue into training permission. Ruff passes for all new code.

A separately implemented standard-library reviewer reconstructs every2,242
photo record, every per-role license/overlap count, all240 consumed-source
bindings and all local original hashes. It matches the complete private rows.
A second execution of its **public-only path** matches all public metadata counts
without opening private photo pairings. These are local implementation checks;
outside human reviewers remain0.

| Guarded phase | Duration | Sampled process-tree RSS lower bound | Result |
| --- | ---: | ---: | --- |
| Metadata acquisition | 20.209s | 48,152,576 bytes | exit0 |
| Complete inventory | 1.577s | 437,157,888 bytes | exit0 |
| Full separate reconstruction | 1.314s | 576,290,816 bytes | exit0 |
| Public-only reconstruction | 1.039s | 505,167,872 bytes | exit0 |

Each guarded phase has zero sampling errors and no guard event, within900seconds,
6GiB sampled process-tree RSS and1536MiB available RAM/disk reserves. Sampling
does not guarantee a true instantaneous peak. The small queue/export steps are
separate short local operations. The252,907,541-byte annotation archive has
SHA256 `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268`.
Its object ETag/MD5 and size were observed over verified HTTPS; this is not a
publisher-signed SHA256 statement. The official custom-host HTTPS address failed
certificate verification, and the tested GCS alternatives were unavailable. The download
used the same S3 bucket through a working certificate-verified path; no TLS bypass.

The [public aggregate package](evaluation-panels/image-source-rights-v19.json)
binds source/program/test hashes, both actual review receipts, resource outcomes
and queue counts. The [standalone metadata summary](evaluation-panels/image-source-rights-v19-summary.json)
supports public-only reproduction. Raw photos, new photo IDs/pairings, private
queue/inventory rows, neural maps, voices and learned weights are not published.
GitHub Actions remains disabled: local checks and direct pushes only. GCP stays
last, Stripe is unchanged, and the preserved voice calibration still has SHA256
`912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf`.

## Reproduce the metadata findings locally

Use Python with `requests` and `psutil`; no models, photos, account credentials,
GitHub workflows or cloud resources are needed. Use fresh paths. The acquisition
is about253MB and downloads pinned metadata/terms only:

```sh
PYTHONPATH=backend python -m scripts.image_rights_runtime_v19 scripts.image_source_rights_v19 /tmp/new-photo-acquire-monitor acquire /tmp/new-photo-metadata
PYTHONPATH=backend python -m scripts.image_rights_runtime_v19 scripts.review_image_rights_v19 /tmp/new-photo-review-monitor /tmp/new-photo-metadata docs/evaluation-panels/image-source-rights-v19-summary.json /tmp/new-photo-review.json
```

The public-only receipt explicitly leaves private consumed-source verification
unassessed; it does not pretend to verify files it cannot access. Report the
commit, environment, result and any failure through the existing evaluation issue
form. No person has been contacted or recruited by publishing these instructions.

## Primary sources and attribution

- Hannes Mareen and colleagues, TGIF/TGIF2, publisher
  [release atfc5c557](https://github.com/IDLabMedia/tgif-dataset/tree/fc5c557bd1a2d49a5dabc3676a54cfc44798f4ea),
  [README](https://github.com/IDLabMedia/tgif-dataset/blob/fc5c557bd1a2d49a5dabc3676a54cfc44798f4ea/README.md)
  and [CC BY-SA4.0 declaration](https://github.com/IDLabMedia/tgif-dataset/blob/fc5c557bd1a2d49a5dabc3676a54cfc44798f4ea/LICENSE.txt).
  Our aggregates and source-grouping methods are additions; they do not override
  the underlying photo licenses or establish all upstream rights.
- COCO Consortium, [pinned terms](https://github.com/cocodataset/cocodataset.github.io/blob/5e1c4da72464b1c6f068df0c02c91e3000ea62c4/dataset/termsofuse.htm)
  and [download/source-host documentation](https://github.com/cocodataset/cocodataset.github.io/blob/5e1c4da72464b1c6f068df0c02c91e3000ea62c4/dataset/download.htm).
  The annotation/website grant is distinct from the photograph copyrights.
- Flickr, [current terms](https://www.flickr.com/help/terms/). Metadata does not
  identify every photographer or resolve all contractual/other rights.
- Creative Commons, [AI-training license guidance](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/).
  This project's metadata filter is conservative; the guidance distinguishes
  copyright permissions, license conditions and jurisdiction-specific exceptions.

Targeted image learning and fresh confirmation remain unfinished. Difficult
audio, lip sync/general video, writing, calibrated combination and outside human
participation also remain open parts of the full improvement program.
