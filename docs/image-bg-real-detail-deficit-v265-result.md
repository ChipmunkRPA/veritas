# The local detail cue responded more strongly to ordinary blur

The replay processed 180 views across 12 source groups with zero model calls. Every view retained the complete AI edit footprint for the paired analysis. The source pixels and both JPEG qualities reuse the same 12 groups. Two replays produced identical aggregate bytes. [Frozen method](image-bg-real-detail-deficit-v265-method.md) · [Exact aggregate](evaluation-panels/image-bg-real-detail-deficit-v265-complete.json)

| Encoding | LaMa edit | Untouched original | Tone | Blur | Copy-move |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source | 0.468 | 0.536 | 0.536 | 0.661 | 0.488 |
| JPEG 75 | 0.470 | 0.527 | 0.533 | 0.612 | 0.489 |
| JPEG 90 | 0.481 | 0.536 | 0.532 | 0.639 | 0.486 |

The table gives mean concordance between tiles touching the fixed AI footprint and remote tiles. A value of 0.5 gives chance ordering. The cue ranked LaMa edits below chance in all three encodings. Ordinary blur had higher concordance even when scored against its own changed-pixel footprint, from 0.617 to 0.665. The LaMa edit produced a higher mean response than every ordinary edit at the same site in just 1 of 12 source views and none of the JPEG views.

The sign and ordinary-control results show that a local fine-detail deficit cannot support regional LaMa detection in this comparison. Keep this cue out of customer reports. A more useful regional method needs to discriminate between object removal and ordinary processing on separately sourced images. [Earlier matched image comparison](image-bg-real-backup-jpeg-v264-result.md)
