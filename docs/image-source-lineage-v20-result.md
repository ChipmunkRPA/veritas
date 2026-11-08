# Photo attribution and exact pixel lineage: 397 matches, 68 differences

**The source-photo checks are complete; detector accuracy has not changed.**
All 523 photos in the [v19 rights-review queue](image-source-rights-v19-result.md)
were checked against current public provider metadata. Of these, 465 matched the
recorded photo identity and license. Every one of those 465 then completed the
source-pixel comparison and a separate local replay. There are 397 exact RGB
matches and 68 differing files. No model ran, no training occurred, and no source
or model is automatically admitted to production training by this result.

The [public aggregate package](evaluation-panels/image-source-lineage-v20.json)
contains the counts, source/program commitments, actual local review receipts,
resource profiles and primary-terms snapshot hashes. It contains no photograph,
new photo IDs/pairings, individual photographer record, neural map or weight.

## Attribution: all 523 planned requests retained

The fixed local plan was saved before the full audit, with SHA256
`50d7f0ea272a98e943dd94df8257b5d1effaeddfa07a9c61aa0b610c43f64d68`.
Every queued source was requested once, in fixed order, without retries or
substitutions. Requests start at least 0.25 seconds apart. Bodies and errors are
retained privately. The audit uses Flickr's public oEmbed endpoint and documented
short-photo URLs; it does not use account credentials or upload a customer file.

Matching requires the exact photo ID, server and image secret from the historic
COCO reference; a consistent author/profile/photo-page identity; photo response
type; and agreement with the recorded BY/BY-SA 2.0 license ID and URL. Provider
metadata establishes an observed attribution correspondence, not independently
verified photographer identity, camera authenticity or complete rights clearance.

| Publisher source role | Planned | Metadata matched | Unresolved |
| --- | ---: | ---: | ---: |
| Training | 420 | 373 | 47 |
| Validation | 79 | 69 | 10 |
| Testing | 24 | 23 | 1 |
| Total | 523 | 465 | 58 |

There are 506 HTTP 200 and 17 HTTP 404 responses. The 58 unresolved cases comprise
17 unavailable records, 35 changed license records, three video responses and
three link responses. **Changed does not mean uniformly restricted:** seven of
the changed records still show BY/BY-SA schemas, including one BY 4.0 record.
Two others have government/public-domain labels requiring separate review.
The remaining 26 show NC, ND or all-rights-reserved schemas. The strict audit is
not silently expanded after seeing these results, and is not a legal conclusion
about the validity or irrevocability of historic grants.

All 523 underlying Flickr photo IDs are distinct and none crosses publisher
roles. The previously discovered category repetitions and misleading FLUX
original CSVs remain explicit; their filenames are not used to assign roles.
A separately implemented, network-free reviewer re-parses all 523 saved bodies,
reconstructs every accepted attribution field, checks all request/source bindings
and reproduces the complete counts. It passes. This is a local reconstruction,
not participation by an outside person or an independent remote observation.

## Pixel lineage: all 465 comparisons complete

The recorded local source plan has SHA256
`80f5f93730677025fe8a1099e11a144cd060173358b2c2b26fd7b5c1f8984af6`.
It was saved before the first source-photo download. A deterministic shared
category is selected per photo from the base and actual generation metadata;
selection uses no detector response or quality score. Original archives are
streamed one publisher role at a time, retaining only the selected PNGs.
The larger Flickr JPEG is retrieved through the exact matched static reference.

Native RGB is decoded without implicit EXIF orientation changes. The only
permitted geometry is the inspected publisher's aspect-ratio rotation rule,
followed by the full image or its explicit metadata crop. No searching over
arbitrary rotations, offsets, tolerances or perceptual similarity is allowed.
An exact match requires every compared RGB byte to agree.

| Publisher role | Compared | Exact RGB match | Differing pixels |
| --- | ---: | ---: | ---: |
| Training | 373 | 317 | 56 |
| Validation | 69 | 61 | 8 |
| Testing | 23 | 19 | 4 |
| Total | 465 | 397 | 68 |

All 465 originals have the full-source geometry; none requires a crop or rotation
in this panel. Every original and source JPEG was retrieved and decoded; there
are no skipped or unresolved pixel comparisons. A pixel mismatch can reflect
content changes, server reencoding or historic decoding differences. It is not
proof of AI manipulation, fraud or loss of copyright permission.

The separate reviewer reconstructs source/category/crop bindings from the pinned
publisher metadata and verifies all retained file hashes. It implements geometry
separately, using integer aspect-ratio arithmetic and explicit row/column
construction. Pillow and OpenCV decode both files for every case: all **930
dual-decoder comparisons** match exactly. All 465 complete geometry/metric
replays reproduce, including every mismatch, with no failed case.

These checks replay retained bytes; they do not independently reacquire the
remote archives. Archive ETags and sizes are observational object bindings. The
stream can stop once all selected members are found; a full archive SHA256 or
final gzip-tail verification is not claimed. The original photos are dataset
source controls, not independently authenticated raw-camera records. Photographer
account separation and near-duplicate/burst-image grouping are not established by
distinct photo IDs and remain part of the prospective training/split review.

A private source index retains all 465 outcomes, per-photo attribution, original
and JPEG hashes and the remaining training-use requirements. It does not discard
the 68 mismatches or turn the 397 matches into training permission. Only 19
previously unscored test-source photos have both strict metadata and exact pixel
correspondence. That does not meet the earlier 48-source confirmation design;
additional permitted sources remain necessary. No neural confirmation test was
run or consumed by this source audit.

## Generator-output rights remain a separate requirement

The original-photo work downloaded no new AI-generated image. A source-photo
license cannot by itself resolve the terms for an edited generator output.

Adobe's current dated [generative-AI terms, section 3.3](https://www.adobe.com/cc-shared/assets/pdf/legal/servicetou/adobe-generative-ai-product-specific-terms-en-us-20260423.pdf)
restrict using generative outputs to create, train, test or improve AI/ML systems.
Black Forest Labs' current [licensing guidance](https://help.bfl.ai/articles/9272590838-self-serve-dev-license-overview-pricing)
calls for a separate synthetic-data license for training on outputs. Its
[noncommercial model license](https://bfl.ai/legal/non-commercial-license-terms)
also contains conditional output-use language and restrictions; a generic
commercial-output statement is not sufficient clearance for this training use.

Those current documents do not establish which contract governed each historic
TGIF image or adjudicate prior research. For new research datasets, Firefly training/testing
and FLUX-output training stay deferred pending release-specific rights. SD2
output/data permission and exact positive edit/control pair reconstruction still
need to be checked before a new supervised experiment. No paid license was
purchased, no vendor was contacted and no generator checkpoint was downloaded.

Primary documents were captured privately with HTTP 200 status and SHA256
commitments in the public packet. Their full text is not republished. Photographer
attribution follows the scope described in Flickr's
[attribution guidance](https://www.flickr.com/services/developer/attributions/)
and [license-schema documentation](https://m.flickr.com/services/api/flickr.photos.licenses.getInfo.html).
The dataset remains credited to Hannes Mareen and colleagues, TGIF/TGIF2,
[release fc5c557](https://github.com/IDLabMedia/tgif-dataset/tree/fc5c557bd1a2d49a5dabc3676a54cfc44798f4ea),
under its declared CC BY-SA 4.0 terms, with the underlying photo grants separate.

## Local verification and reproduction limits

All 49 authored tests pass without skips. They exercise wrong photo/owner/license
identities, untrusted URLs, exact crop offsets, one-pixel differences, orientation
refusals and both actual image decoders. Ruff passes for all ten new code/test
files. The tests use authored images and metadata; they require no photos,
weights, account credentials, workflow or live network request:

```sh
PYTHONPATH=backend python -m pytest -q -o addopts='' backend/tests/test_image_attribution_v20.py backend/tests/test_image_pixel_lineage_v20.py backend/tests/test_image_lineage_review_v20.py
```

Actual private-panel reproduction additionally requires the retained source
plans and separately acquired permitted photos. It is not advertised as a fully
public photo panel. Contributors can review the public code/count commitments,
run the authored checks, or propose their own permitted source-separated panel
through the [existing evaluation contract](independent-testing.md).

| Guarded phase | Duration | Sampled tree RSS lower bound | Result |
| --- | ---: | ---: | --- |
| Attribution acquisition | 134.496s | 43,270,144 bytes | exit0 |
| Attribution reconstruction | .264s | 1,507,328 bytes | exit0 |
| Training originals/JPEGs | 640.188s | 96,649,216 bytes | exit0 |
| Training pixel replay | 13.161s | 159,973,376 bytes | exit0 |
| Validation originals/JPEGs | 97.322s | 103,448,576 bytes | exit0 |
| Validation pixel replay | 2.390s | 132,464,640 bytes | exit0 |
| Testing originals/JPEGs | 81.044s | 102,285,312 bytes | exit0 |
| Testing pixel replay | 1.058s | 126,287,872 bytes | exit0 |

All guarded phases stayed within 900 seconds, 6 GiB sampled tree RSS and
1536 MiB available RAM/disk reserves, with no guard event or sampling error.
Sampled RSS is a lower bound, not an instantaneous peak guarantee. Preparation
probes and short export/terms-capture steps are separate. An initial relative-path
planning error is retained with its original program bytes; the valid successor
normalizes the path and uses a new output folder. It occurred before any source
photo download. Unsupported oEmbed-query/parser probes also remain retained.

GitHub Actions stays disabled; verification is local and publication is a direct
push. No app detector, public confidence claim, model-name filtering, prepayment
behavior, rollback asset or GCP resource changed. The voice-calibration SHA256
remains `912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf`.

Outside human reviewers remain zero. New supervised regional learning, additional
confirmation sources, difficult audio, lip sync/general video, writing and
cross-media calibration remain unfinished parts of the full improvement goal.
