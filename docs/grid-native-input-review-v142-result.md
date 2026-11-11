# Complete fresh input bank passes independent replay

All sixteen originally selected GRID source archives have complete acquisition
records. Both preselected clips from every source have complete native banks:
**32 clips, 2,400 native pictures and 1,524,736 canonical mono 16-kHz samples**.
No original slot was replaced or dropped. The original v131 time-limit stop
remains failed; its completed first archive was admitted through the independent
v134 preflight, with subsequent acquisition recorded separately.

The separately frozen v142 worker completed in **8.46 seconds**, exit zero,
with no guard or sampling errors. Maximum sampled descendant RSS was
54,067,200 bytes (51.56 MiB). The unchanged data-stage limits were 900 seconds,
1.5 GiB sampled RSS, 3 GiB initial RAM/disk and 1.5 GiB running reserves.
See the [actual measurement and resource profile](evaluation-panels/grid-native-input-review-result-v142.json)
and [pre-execution method](grid-native-input-review-v142-method.md).

The worker independently verified selected MPG byte counts, SHA commitments and
ZIP-member CRCs without decoding MPG or reopening full archives. Pillow decoded
all saved PNGs and reproduced their RGB-byte commitments. Python's WAV reader
independently reproduced every stored float32 audio sample. Exact rational
calculations checked frame indices, native PTS, audio origins, sample ranges,
continuity and endpoint agreement allowing the declared resampling quantization.
All 32 clips remain available; unavailable inputs would have stayed unavailable.

There were **zero exact complete-picture-sequence duplicate groups and zero exact
PCM duplicate groups**. Joining those content links with the original source
slots produced sixteen components, each containing that source's two clips.
This establishes known content/source grouping only. It does not prove distinct
people, unrelated recording ancestry or absence from pretrained-model training.
All model evaluation variants from a linked component must remain together.

Thirty-six authored review tests, Ruff and diff checks passed before the frozen
contract was published and executed. An additional internal code review checked
the receipt chain and resource/clock handling. No outside reviewer participated.

This validates input consistency, not detector accuracy or achieved lip sync.
The next transfer comparison still needs generated edits, matched ordinary
controls and the actual serving pipeline baseline. The public package contains
code, aggregate counts and whole-artifact commitments; media, individual hashes,
identities, pairings, clocks and vectors remain private. No detector/default,
customer confidence, GitHub Actions, outreach, payment or deployment changed.
