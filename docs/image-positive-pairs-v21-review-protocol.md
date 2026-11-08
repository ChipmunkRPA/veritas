# Separate source-pair review before regional learning

The [paired-data acquisition](image-positive-pairs-v21-protocol.md) is still
running. This supplement freezes a separate local review before its first
full-project execution. It does not alter the 397 sources, generator variation,
pixel-truth rules, acquisition code, failed phase, model or serving confidence.

The [review freeze](evaluation-panels/image-positive-pairs-v21-review-freeze.json)
binds the reviewer, its monitor, tests, decoder versions and unchanged private
source plan. All **19 authored tests pass locally**. They include a complete
397-pair synthetic integration case and refusals after an altered truth image,
ordinary-control digest or deliberately omitted available asset. These authored
examples establish program behavior; they are not detector-quality results.

## What the separate review checks

- Every archive checkpoint, predecessor, private index digest, completion seal,
  selected member, phase resource record and stored HTTP range receipt must
  agree. A later checkpoint cannot silently replace or drop an earlier member.
  Repeated byte-range digests must agree across the entire archive chain.
- Every executed phase must be bound by the execution journal. The successful
  copied seed is counted separately from new execution. The old phase with an
  unexplained sampling error must remain retained and failed.
- The selected checkpoint entries must bind the actual edit and alpha PNG used
  in qualification. Every case remains in the complete 397-source denominator;
  an available input cannot be silently discarded as missing.
- Pillow difference/channel operations and histograms independently reconstruct
  the whole changed-pixel mask and confirm that changes stay inside the declared
  compositing support. Pillow and OpenCV independently decode every available
  source/edit RGB pair, and original RGB must match the earlier lineage digest.
- Scalar binary64 lookup operations reconstruct the frozen tone recipe. Integer
  alpha composition reconstructs both ordinary-control RGB digests. All 256
  alpha values and all 768 tone lookup entries are checked in authored tests.
- The retained truth PNG is checked both by its byte digest and its complete
  decoded pixel content. Individual case failures are retained in the result;
  a failed review cannot be reported as successful.

The tone reference deliberately follows the original floating-point operations.
An ideal rational rewrite differs at some half steps and would not reproduce
the declared control. The Gaussian blur kernel and mask mode conversion still
share Pillow with the producer; they are not claimed to be independent kernels.

## Resource monitoring and preserved failures

The new explicit review-only monitor retains the same 900-second, sampled
6-GiB process-tree RSS limit and 1536-MiB available RAM/disk reserves. It also
checks reserves before launching a worker. A missed live process sample stops
the phase; a verified terminal transition is retained with the actual exit
status. A failed result or existing output folder is not automatically retried
or overwritten. No neural inference or fitting runs in this review.

The first authored run passed 13 tests and failed two actual-process tests
because the restricted environment denied process monitoring. Both failure
receipts and that test report remain retained. An explicitly authorized host
run passed those 15 tests without changing the monitor. Four full-denominator
integration tests were then added, and the final 19-test host run passed.
The earlier restriction failures are not reclassified as successful execution.

## Limits and next step

Stored HTTP receipts are audited for consistency; their remote payload bytes
are not independently downloaded again by this reviewer. ETags are not signed
checksums, no complete archive digest is claimed, and imported gzip indexes
disable whole-stream CRC checking. Pixel comparisons bind the publisher-labeled
alteration, not human authorship or camera authenticity. This is a second local
implementation, **not outside human review** or independent model evaluation.

After acquisition and primary qualification finish, run the separate review
under its bound local monitor and preserve the actual complete result. Only
then prepare source/photographer-connected learning roles and a separately
frozen supervised experiment. No training admission, new confidence score or
default change follows from this preparation alone.

Raw photos, source pairings, masks, gzip dictionaries and learned weights stay
private. Publish aggregate results and commitments only. GitHub Actions remains
disabled; validation is local and publication uses direct Git pushes. GCP is
still deferred.
