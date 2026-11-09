# Fresh image source attribution: prospective expansion

This is a metadata-only source-collection protocol, not an evaluation result or
detector improvement. It expands the possible source pool beyond the 58 cached
photos described in the [earlier feasibility review](image-fresh-source-route-v34-v35.md).
No images or generator weights are downloaded in this stage.

## Fixed selection before current-provider requests

The already verified COCO 2017 annotation archive contains 123,287 image records.
Exclude the union of every source ID in the existing 705-record source graph and
the earlier 523-record attribution queue: 763 distinct IDs. Retain only historical
CC BY 2.0 or CC BY-SA 2.0 records with recognized Flickr static-photo URLs.
This leaves 31,064 distinct eligible photo IDs.

Rank by SHA-256 of `fresh-image-attribution-v37/20261108/{COCO source ID}` and take
the first 256 distinct Flickr photo IDs. Selection never uses detector responses,
image appearance, model-generated edits or the success of a current metadata
request. No replacements or implicit retries are allowed.

## Current attribution and licence correspondence

Run eight explicit phases of 32 public Flickr oEmbed requests. Each request has
a 100,000-byte body limit, 10/20-second connection/read timeouts and at least
0.25 seconds between request starts. Retain response bytes, observation time,
source binding and every unresolved result.

A match requires the exact historical photo/server/secret, a named profile,
consistent canonical photographer/photo URLs and current correspondence to the
recorded BY or BY-SA 2.0 identifier and URL. Reconstruct successful rows from the
retained response bytes during review. A changed licence, missing photo,
conflicting owner, timeout or unsupported response remains unresolved.

Report matched creator keys that overlap any known creator in the 705-record
inventory separately. Unlinked canonical profile keys are not proof of distinct
people. The earlier inventory still has 14 unresolved creator identities.

## Resource and execution evidence

Each phase has a separate process monitor: 900-second limit, 6 GiB sampled tree
RSS limit, 1.5 GiB available-RAM/disk reserves and initial 6 GiB RAM/3 GiB disk
headroom. Preserve terminal exit status, observations, errors and resource-journal
hashes. Require successful preceding phases and complete ordered denominators.
Never overwrite or resume a failed phase by skipping its failures.

The first planning attempt could not inspect the child process under restricted
Mac permissions; the monitor terminated it with exit -15 and retained a failure
receipt. A new, explicitly separate planning directory then completed with host
process inspection enabled. Its 1.33-second run had no monitor error or resource
crossing; the largest sampled worker RSS was 503,169,024 bytes, not a guaranteed
peak. The failed attempt is not relabelled as successful.

Before any current-provider request, publish and remotely verify the
[aggregate freeze](evaluation-panels/image-fresh-attribution-v37-freeze.json).
Bind the private execution receipt to the exact plan hash and remote freeze blob.
Local validation: 36 focused tests cover the new collection rules and the existing
photo/owner/licence correspondence checks; Ruff passes. These tests do not
establish source independence or detector performance.

## Admission remains separate

This stage assigns no fit, calibration or test roles and performs no neural calls.
Current source attribution is not raw-camera authentication, complete permission
for every downstream use, or assurance against unknown pretraining overlap.
Source pixels, exact/near duplicates, creator aliases, orientation and generator
component rights still require review before a controlled-generation study.
Freeze that study's roles, masks, prompts, seeds, controls, formats and failure
denominators before project-source generation or detection.

Publish code, counts and hashes only. Keep response bodies, photo IDs, creator
profiles and source pairings private. This work does not change the running v26
training, its held-out roles, serving defaults or public confidence claims. No
external person is contacted, no workflow runs, and no cloud resource is used.
