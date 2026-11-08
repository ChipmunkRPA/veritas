# Explicit successor for the private index-header review

The complete v21 acquisition and primary qualification finished: **397/397**
source-relative pairs passed the declared pixel checks. The first separate v21
review then stopped before pixel replay because it required every exported gzip
index to contain at least one seek point. Two complete small mask archives have
valid35-byte indexes containing zero points. Both retained byte digests match
their committed checkpoint records. This was an unsupported reviewer assumption,
not a changed source image or an unexplained acquisition failure.

The old reviewer, source freeze, failed resource receipt, worker traceback and
stopped local sequence remain unchanged. The old full review still **failed**;
it is not relabeled as passing. No model forward or project fit ran after that
stop. The earlier acquisition sampling failure also stays retained and failed.

## Evidence for the correction

The pinned reader's
[zran.c at d87b55d](https://github.com/pauldmccarthy/indexed_gzip/blob/d87b55d872a6120423e33acfdf7e0a1c836f7816/indexed_gzip/zran.c)
explicitly permits an imported index with zero points. Its exported version1
header contains magic/version/flags, compressed and uncompressed sizes,
spacing, window size and point count. The format currently writes native-endian
integers; this local successor explicitly requires the pinned little-endian
environment. Source blob: `f57601cdb5f522a85465b56bda7b4b5293cb21e7`.

An authored small gzip stream exports exactly35 bytes and zero points. Importing
that index, seeking forward/back and reading the entire original authored stream
works exactly. Import still disables gzip CRC checking, as already disclosed;
this correction does not reinterpret that limitation. A separate larger random
authored stream checks a real populated index too.

## Bound successor, unchanged scientific rules

The [successor freeze](evaluation-panels/image-positive-pairs-v22-review-freeze.json)
binds the new reviewer/monitor/tests and the preserved original failure. It
changes only index-format validation and explicit failure-preservation checks.
It does not change the397 sources, reference masks, control recipes, image
decoding, source containment, complete truth replay, acceptance criteria,
neural method or customer confidence.

The successor requires the complete index digest and a matching35-byte header:
`GZIDX`, version1, zero flags, exact frozen archive size,8-MiB spacing,
32768-byte window and a point count matching the checkpoint. A zero-point index
must be exactly35 bytes; a populated index must include an entry payload. It
still audits the complete checkpoint/range/selected-member chain and every
source pixel/control proof. This is a header/commitment audit, not an independent
decompression of each retained private index or a signed whole-archive checksum.

All13 additional local tests pass. They cover real empty/populated exports,
empty-index import/seeks, resealed invalid headers, the full397-pair authored
pixel/control replay, refusal of changed pixel evidence and actual successful/
failed short workers. The first12-test report is also retained; the populated
index test was then added. These tests do not claim real-data review success.

Publish and verify the successor before a new explicitly recorded full review.
Only a completed passing successor may unblock source grouping and the authored
backward probe. Preserve its actual result under its own version; do not rerun
the failed original sequence. No source data, pairings, raw index bytes or
weights are published. GitHub Actions stays disabled, with local checks and
direct pushes only.
