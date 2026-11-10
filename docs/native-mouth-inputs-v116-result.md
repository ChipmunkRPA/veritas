# Native mouth-render input bank: completed preparation

All eight frozen decoder-only batches completed successfully. The bank retains
all **48 source recordings** and their original sentence-connected study roles.
Forty-seven sources supply an existing 20-picture edit plus two context pictures
before and two after: **1,128 native pictures**, including 940 intended edit
pictures. The short source remains explicitly unsupported without invented
frames or a replacement source.

For every retained picture, native decoded RGB bytes and integer timestamps
match the original v92 qualification. Lossless PNGs saved privately on Seagate
also match after separate Pillow decoding. Source media and corresponding
private speech-state files were rehashed. Exact clocks, source/frame positions,
picture files and voice-derived state bindings remain private. The
[public result](evaluation-panels/native-mouth-input-result-v116.json) contains
counts and whole-artifact commitments only.

Each batch exited zero, with no guard or sampling error. No neural model ran.
The fixed decoder-only budget was retained; neural work still needs its separate
6-GiB initial RAM admission. These input checks do not establish that any source
has a uniquely usable face, that a mouth renderer achieves synchronization or
that a detector recognizes deepfakes. All face-geometry and generator-conditioning
qualification flags remain false.

The next [renderer experiment](native-mouth-renderer-v117-method.md) uses the
same source picture and seed with aligned, repeated, 320-ms earlier and zero
speech conditioning. Its [isolated runtime context](native-mouth-runtime-v118-method.md)
preserves the initial preparation error and uses the already retained research
dependencies, without changing the application environment. It must pass before
the wider native clip construction and articulation detector comparison.

The 14 authored input checks pass, covering exact 20-frame/context support,
unavailable audio, short-source preservation, unchanged source planning, worker
routing and separate data/neural admission limits. The runtime/renderer checks
are separate software tests, not successful neural execution or new accuracy.
Serving models, confidence and rollback assets remain unchanged. No outreach,
GitHub Actions or production deployment was performed.
