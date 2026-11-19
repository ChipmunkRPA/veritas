# Face evidence on HEVC clips with a 2,112-sample AAC lead-in

The optional short face comparison now accepts a verified 2,112-sample
AAC lead-in on the same 4.8-second, 720p25 HEVC source profiles as the
previous AAC comparison. The submitted MP4 must have one AAC-LC track
whose explicit edit, native packet table and decoded start agree with the
picture span. The saved file must describe exactly 120 native frames on
the 90 kHz clock. The evidence records the native clock and uses original
before, during and after frames. These observations carry zero weight in
the overall AI assessment. [Earlier timing feasibility](video-apple-aac-priming-feasibility.md)

## Verification

Seven previously qualified picture roles were retained in each of three
Apple Core Audio encoding layouts. The experiment copied the original
HEVC packets and added a known timing waveform via an explicit MP4 edit.
The same short face checks were also applied to the original comparators.

| AAC layout | Valid samples | Decoded trailing samples beyond the edit | Exact face and coverage comparison |
| --- | ---: | ---: | ---: |
| 44.1 kHz mono | 211,680 | 224 | 7/7 |
| 48 kHz mono | 230,400 | 960 | 7/7 |
| 48 kHz stereo | 230,400 | 960 | 7/7 |

Both known swaps kept the exact ranges **3.080–3.240 s** and
**3.880–4.000 s** in each layout. Ordinary blur still produced the existing
**2.920–3.040 s** highlight. The unblurred swap stayed unassessed. A measured
ordinary blur supplied a no-match control, and an unchanged clip and hard
cut supplied separate controls. These are seven picture roles with repeated
soundtracks, with no new population-level sensitivity or false-alarm estimate.

The native audit matched 3,240 picture clocks, compressed packets and
RGB frames across 27 deliveries with an exact-picture comparator. The
additional native Apple passthrough control lacked the required edit.
The largest timing-marker difference was 18 samples at 48 kHz. Across the
21 qualified files, a separate replay covered 84 measured views and 27
before, during and after frames from the nine localized records. Seven
controls did not qualify under the extended profile. A file carrying
only the Apple timing label still qualified under the earlier profile
because its actual MP4 edit described a supported 1,024-sample lead-in.
[Apple's explicit edit representation](https://developer.apple.com/documentation/quicktime-file-format/using_track_structures_to_represent_encode_delay_explictly)

The prepared fixtures support this specific clock and face-evidence scope.
The audio test uses authored timing tones. Wider native Apple exports,
other playback layouts, lip sync and voice forgery require separate tests.

## Local app replay

The local test ran 19 simulated USD 9.99 checkouts, with zero analysis
jobs before payment. Chrome opened all nine qualified results, including
one historical video-only file and one historical 1,024-sample AAC file.
Each result exported a matching PDF and JSON, and all nine signed packages
verified against their original MP4. Chrome showed the actual before,
during and after pictures and jumped to the reported start. Six unsupported
timing controls stayed outside the optional comparison. The existing
combined assessment and all recorded soundtrack measurements matched in
four paired runs with and without the optional face check.
[Verification receipt](evaluation-panels/video-apple-aac-face-evidence-2026-11-19.json)
