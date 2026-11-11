# Local codec comparison for repeated-audio evidence

All 256 planned observations decoded and reached both repetition checks.
Each channel contains 48 ordinary copies and 16 untouched genuine
controls from the same consumed GRID pilot. The known-copy count
requires overlap in both surviving sample ranges.
[Method](grid-audio-repetition-codecs-v170-method.md) · [Actual result](evaluation-panels/grid-audio-repetition-codecs-complete-v170.json)

| Channel | Exact copy witness | Similarity copy witness | Either witness | Untouched with any reported span |
| --- | ---: | ---: | ---: | ---: |
| PCM | 48/48 | 25/48 | 48/48 | 0/16 |
| AAC 64 kb/s | 4/48 | 28/48 | 29/48 | 0/16 |
| MP3 64 kb/s | 0/48 | 24/48 | 24/48 | 0/16 |
| Opus 48 kb/s | 0/48 | 29/48 | 29/48 | 0/16 |

The similarity check recovered evidence after every lossy encoding type.
It missed 63 of 144 known lossy copies and reported one additional MP3
span that failed the known-copy overlap check. The two methods together
found 82 of 144 known lossy copies. The PCM and codec arms share the same
16 source recordings, so those observations are dependent.

The app's decoder verified one continuous clock for every encoded input.
MP3 kept a 1,105-sample container origin. The AAC output included 480
decoded samples beyond the exact declared duration, which the comparison
excluded. The ranges in the test use decoded-sample coordinates.

The data-only worker completed in 33.76 seconds and reached 93,749,248
bytes of sampled descendant RSS. These checks offer bounded inspection
evidence of repeated samples. Fresh genuine recordings, genuine donor
splices, known AI inserts and noisy variants are still needed to assess
how often the cue points to ordinary or unrelated sound.
