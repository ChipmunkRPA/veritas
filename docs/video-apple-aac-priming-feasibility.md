# Apple AAC timing in an HEVC MP4

This local feasibility check on November 19, 2026 used one authored 4.800-second tonal waveform and one existing local 1280×720, 25 fps HEVC video fixture. macOS 26.5.2 Core Audio encoded the waveform as mono 48 kHz AAC-LC at a nominal 128 kb/s. Apple’s `afinfo` reported 230,400 valid samples, 2,112 leading encoder samples and 960 trailing remainder samples across 228 AAC packets. The MP4s were FFmpeg test remuxes. They represent four clock treatments of the same encoded audio and video.

| MP4 clock treatment | AAC stream duration at 48 kHz | Samples output by unbounded FFmpeg decode |
| --- | ---: | ---: |
| Ordinary stream copy | 233,472 | 233,472 |
| Stream copy with the Apple gapless tag copied to file-level metadata | 233,472 | 233,472 |
| Stream copy with a 2,112-sample leading edit | 231,360 | 231,360 |
| Leading edit plus a 64-sample final packet duration | 230,400 | 231,360 |

All four MP4s retained the same 228 coded AAC packets. Against the video seed, each retained all 120 coded HEVC packet hashes and timestamps, the HEVC decoder configuration, all 120 decoded RGB frame hashes, and all 120 independently probed native picture timestamps. Video used the 90,000 Hz track clock throughout. The metadata-copy case moved the gapless tag into file-level metadata and left FFmpeg with the same 4.864-second audio timeline as ordinary copy.

The final MP4’s movie edit starts at AAC sample 2,112 and presents 432,000 movie ticks at 90,000 Hz. FFprobe observes the leading skip and the final packet’s declared 64 samples. FFmpeg 7.0.2 still outputs the entire terminal decoded block, adding 960 samples beyond the declared interval. Capping the 48 kHz decode to 230,400 samples reproduced the matching slice of the full decode exactly. Under the same signed 16-bit conversion, resampling that capped signal produced 76,800 samples at 16 kHz and matched independent resampling of the PCM slice byte for byte. Core Audio separately decoded the original M4A and final MP4 to identical 230,400-sample PCM outputs.

The bounded case supports an experiment on exact audio-clock qualification. Before extending app coverage, require independently consistent track edits, packet clocks, one declared presentation extent and sample counts. Apply any established tail limit before resampling, and include unsupported timing and ordinary controls. A gapless tag or the historical 2,112-sample convention alone cannot establish an uploaded file’s valid sample range. This check tested one authored waveform, one video, one Apple encoding configuration and one explicit remux recipe. It measured clocks and decoded media equality only. It supplies no lip-sync detection rate, natural camera-export coverage or authorship evidence. [Apple’s explicit encoder-delay representation](https://developer.apple.com/documentation/quicktime-file-format/using_track_structures_to_represent_encode_delay_explictly), [Apple’s historical encoder-delay convention](https://developer.apple.com/documentation/quicktime-file-format/historical_solution_implicit_encoder_delay)

The [count-only receipt](evaluation-panels/video-apple-aac-priming-feasibility.json) contains measured clocks and sample counts.
