# Source-bound SD2 pairs: preparation and fixed acquisition plan

The next regional-learning dataset must bind each generator edit to its actual
source pixels and declared compositing region. This plan qualifies **all 397**
source photos that passed the [current attribution/pixel-lineage checks](image-source-lineage-v20-result.md):
317 publisher-training, 61 validation and 19 testing sources. These publisher
labels are not yet new fitting/calibration/confirmation assignments. No model
forward, fit, confidence update or serving change is authorized by qualification
alone.

The [public freeze](evaluation-panels/image-positive-pairs-v21-freeze.json)
commits the complete private plan and six source/test files before the first
retained project edit PNG. It preserves every planned source, including missing
or invalid references. It does not select cases using detector responses.
The [resource supplement](evaluation-panels/image-positive-pairs-v21-resource-supplement.json)
also binds the unchanged shared guard and serving voice calibration; it changes
no selection or qualification rule.

## Data and permission scope

Use only the publisher-released SD2 splice, bbox selection and generator
variation 0, in the same shared category already chosen in v20. Original photo
bytes, current BY/BY-SA 2.0 attribution and exact source RGB are already bound.
The [IDLabMedia TGIF/TGIF2 release](https://github.com/IDLabMedia/tgif-dataset/tree/fc5c557bd1a2d49a5dabc3676a54cfc44798f4ea)
declares CC BY-SA 4.0 and is explicitly released for forgery-localization training
and evaluation. Retain its attribution and separate underlying photo grants.
This is bounded local paired-data qualification under those declarations, not
a blanket rights warranty or independent camera-authenticity finding.

No SD2 generator weight is acquired or run. This does not claim independent
verification of a now-unavailable original generator checkpoint's license.
Firefly and FLUX-output training remain outside this new acquisition. Their
release-specific contract questions, production-training admission, photographer-
connected source splits and a prospectively frozen learning experiment remain
separate work. There is no automatic promotion from this data plan to serving.

## Exact pair and ordinary-control checks

For every original/edit pair:

1. Verify the planned original, edit and author compositing-mask hashes; retain
   bounded upright PNGs in their native source coordinates.
2. Require matching source/edit shape and mask dimensions. Compute the actual
   changed RGB pixels, keeping the declared selection separate from that truth.
3. Require a nonempty edit and no changed pixel outside the published positive
   compositing alpha. Do not relabel an unsupported pair as an original image.
4. Commit the complete changed-pixel mask. Generate two deterministic ordinary
   controls from the same source and alpha: local tone/colour adjustment and
   Gaussian blur. They use no generative model and receive no positive AI label.
5. Store control recipes and full RGB hashes, generating their pixels in memory
   rather than retaining redundant full PNG copies. Keep every rejection/error
   in the denominator. Independent native/pixel reconstruction is required before
   any learning stage.

The reference is the publisher-labeled inserted alteration, not proof of the
authorship of every pixel in a customer file. JPEG augmentation, when later
defined, must preserve the pre-compression edit coordinates rather than treating
whole-image JPEG differences as newly generated pixels.

## Bounded, resumable archive access

The publisher's training-edit archive is 14,359,902,282 bytes. Its validation and
testing edit archives are 2,172,017,290 and 2,040,575,228 bytes. Whole copies would
unnecessarily consume the Mac's available disk. Instead, one explicit phase reads
exact HTTP ranges and retains only selected PNGs plus a private gzip seek index.

- Bind the remote URL, exact size, ETag and every requested byte range; require
  HTTP 206, exact Content-Range, unchanged ETag and an untransformed complete
  body. No retries or account credentials are used.
- Complete the current tar member, then stop after 240 seconds or 2 GiB of
  transferred ranges. One member/read-ahead can extend that soft boundary.
  A separate outer guard retains the 900-second/6-GiB and 1536-MiB reserve rules.
- Export a private partial index and exact next-tar offset, preserving PAX
  metadata. A new phase requires the predecessor's committed checkpoint,
  unchanged retained bytes, completed seal and successful resource profile.
  A failed phase is not silently restarted or skipped.
- Preserve all range journals, indexes and phase records. Imported indexes
  disable whole-gzip CRC validation in the upstream library. ETags are not
  signed checksums; full-archive cryptographic validation is not claimed.
  Retained selected PNG hashes and independent pixel reconstruction remain
  required.

The unchanged optional reader is
[Paul McCarthy's indexed_gzip](https://github.com/pauldmccarthy/indexed_gzip/tree/d87b55d872a6120423e33acfdf7e0a1c836f7816),
version 1.10.3, under the [Zlib license](https://github.com/pauldmccarthy/indexed_gzip/blob/d87b55d872a6120423e33acfdf7e0a1c836f7816/LICENSE).
Its 338,676-byte arm64 wheel was hash-checked and installed only in a research
target; app dependencies are unchanged. Its decompression-history indexes can
contain source fragments and must never be published as harmless metadata.

## Preparation evidence and remaining execution

All 18 authored local tests pass. They verify complete-range/refusal behavior,
real partial-index continuation across 12 long-name/PAX tar members, changed
checkpoint refusal, exact masks, out-of-region changes and repeatable ordinary
controls. An initial test incorrectly required fewer phases than members;
startup read-ahead produced one completed member per phase. The original test
and failure are retained; the corrected test requires strict progress and all
final bytes. No real pair or model result was used to change a scientific gate.

An actual header/footer probe made two exact requests against the public archive,
transferring 6,218 compressed bytes and decoding no image. The small training-mask
archive was inspected separately to verify the filenames. These are preparation
checks, not evidence that all project pairs are acquired or valid.

The complete 397-source acquisition, qualification, independent review and any
training/confirmation are still pending at this freeze. Source photos, selected
edits, masks, new pairings and learned weights remain private. Use a fresh folder
for every phase. GitHub Actions remains disabled; checks and direct pushes only.
The wider image/video/audio/writing improvement and outside human-review goals
remain active, with GCP deferred.
