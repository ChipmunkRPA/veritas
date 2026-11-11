# Prospective genuine HD video controls

This local pilot measures false alarms on three publisher-documented genuine
recordings. Each recording contributes one speaking excerpt and one paired
ordinary recompression. The recordings and derived clips stay private. This
study uses no face or voice replacement, neural redraw, or customer material.
Three planned recording groups provide an engineering check of serving
coverage, timing and false flags.

| Group | Source and attribution | Publisher file metadata |
| --- | --- | --- |
| 01, Wikimania Nairobi 2025 | *Interview with Jimmy Wales at Wikimania 2025 Nairobi*, DelwarHossain and Rocky. CC BY 4.0, own work. Group with other material from the same event and interview crew. [Pinned source record](https://commons.wikimedia.org/w/index.php?curid=172874828&oldid=1074684101) · [Original WebM](https://upload.wikimedia.org/wikipedia/commons/f/f6/Interview_with_Jimmy_Wales_at_Wikimania_2025_Nairobi.webm) | 1920 × 1080 VP9/Opus, 248.776 s, 115,172,735 bytes. Original upload `2025-08-17T17:07:01Z`. Publisher SHA-1 `5b62001d250c4387edc8007d9e55fdd67c8c9d84`. |
| 02, Sustainable Future workshop, June 2024 | *Interview with participant in the Workshop on Using Wikimedia Projects for a Sustainable Future 1*, Muhammed amine benloulou. CC BY-SA 4.0, own work. Group with other footage from the same workshop. [Pinned source record](https://commons.wikimedia.org/w/index.php?curid=150334682&oldid=958200493) · [Original WebM](https://upload.wikimedia.org/wikipedia/commons/2/28/Interview_with_participant_in_the_Workshop_on_Using_Wikimedia_Projects_for_a_Sustainable_Future_1.webm) | 1920 × 1080 VP9/Vorbis, 104.100 s, 143,370,279 bytes. Original upload `2024-07-14T19:01:58Z`. Publisher SHA-1 `38a67682d92f076d43546be72ade8d4e082eb129`. |
| 03, USGS employee interview compilation | *Working with the USGS MD-DE-DC Water Science Center*, USGS Northeast Region. Credit Natalie Bates, contractor for USGS, and Lonnie Lanham, USGS. The asset page marks the work Public Domain and lists no separate soundtrack restriction. Group all excerpts from this compilation and any known shared people or recording sessions. [Source and credits](https://www.usgs.gov/media/videos/working-usgs-md-de-dc-water-science-center) · [Official MP4](https://usgs-ocapsv2-public-input-media.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/video/Center%20Video.mp4) | Publisher duration 3:49. Official object `HEAD` on 10 November 2026 reports 282,426,173 bytes, last modified `2025-11-04T22:48:13Z`, full-object CRC64NVME `djbo3YUD1Sk=`. Frame size, video and audio codecs, and clock await preflight. |

Commons file sizes, digests and revisions come from the `imageinfo` and
`revisions` responses for the pinned titles. Preserve both Commons creators'
attribution, source and license links and note the ordinary edits in the local
ledger. The ShareAlike condition applies when adapted material is shared.
Verify the USGS checksum with an S3-compatible full-object CRC64NVME check
and save a local SHA-256. The multipart ETag does not provide a file digest.
[Commons metadata API](https://www.mediawiki.org/wiki/API:Imageinfo) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en#s3b) · [S3 object metadata](https://docs.aws.amazon.com/AmazonS3/latest/API/API_HeadObject.html)

Before any scoring, acquire each exact source and check decoded HD dimensions,
picture PTS and cell duration, the audio sample clock including codec start
offset, and source/content overlap with earlier panels. Document the earliest
continuous shot with one visible adult lead speaker, original continuous
speech, and clear audio and image rights. Within that shot, use its earliest
valid frame as the start and the first observed frame boundary at least six
seconds later as the end, up to 7.2 seconds. No clip selection uses detector
outputs. Retain a candidate as unavailable when these checks fail. Keep
recording groups intact if later checks identify a shared source or speaker.

Preserve the native timing through the reference and its H.264 recompression.
Freeze excerpt boundaries, the encoder settings, and the exact source-to-case
picture and audio mappings before scoring. Retain the original decoder-reported
time bases and any measured time offsets, including unsupported cases. Report
tested coverage and false flags by group. The source and control counts support
no inference about detection sensitivity or model-training independence.
[Serving scope](video-in-envelope-transfer-design.md)

## Local acquisition and prefix clocks

Group 01 is held locally on the external drive. Its 115,172,735-byte length
and publisher SHA-1 matched. The recorded whole-file SHA-256 is
`82d47b86fed44d9e3110777dd694466c879b259f36ec6a3b1adb719bd975aa62`.
Stream metadata reports 1920 × 1080 VP9 at 25/1 nominal and average, video
origin zero at time base 1/1000, and stereo 48 kHz Opus with a reported
stream start of −7 ms. Container duration is 248.776 seconds. Per-picture
PTS in a ten-second prefix decoded without seeking form 250 consecutive
pictures from 0 to 9,960 ms, with all 249 increments exactly 40 ms.
The first decoded audio frame starts at 0 ms and has 648 samples. The next
starts at 14 ms and has 960 samples. Its timestamp is half a millisecond
later than the first frame's sample count predicts at 48 kHz. All 498
subsequent intervals in the prefix match the 960-sample increments.
Seeking to timestamp zero before probing returned a first decoded audio
frame at 21 ms, so the candidate preflight must distinguish the source's
decoded audio origin from a seek-dependent packet or stream timestamp.

Group 02 matched its 143,370,279-byte length and publisher SHA-1. Its local
SHA-256 is `df0bf6f5e9129b844610e1b8cfe41a2eb115c9b796acb7efb6707ce5ac73a600`.
The prefix has 300 decoded 1920 × 1080 pictures. Their 1/1000 PTS increments
are 33 or 34 ms. The first two decoded 44.1 kHz Vorbis audio frames start at
13 and 36 ms and contain 576 and 1,024 samples. The first PTS increment
exceeds the first frame's extent by about 10 ms. Other increments have
positive and negative residuals beyond millisecond quantization. The full
decoded audio track has 5,637 frames. Its longest contiguous run consistent
with a single sample origin under the 1-ms timestamp precision spans 155,648
samples, about 3.529 seconds. Group 02 remains in the planned denominator
with audio timing unavailable under this preflight. It can still provide a
visual-only control once its shot is verified.

Group 03 matched its 282,426,173-byte length and full-object CRC64NVME. Its
local SHA-256 is
`b0fe1ca1d257d27fa6898db48b06b4e58640699a9fbd247e63e04cb8af4bc527`.
The prefix has 299 decoded 1920 × 1080 H.264 pictures with constant 1,001-tick
increments on a 1/30000 clock. The first decoded 48 kHz AAC audio frame starts
at zero on a 1/48000 clock. All 453 observed audio frame increments match the
preceding frame's exact sample extent.

The production audio decoder independently processed the first ten seconds
of all three source files with the source SHA-256 verified first. Each
produced 160,000 mono 16-kHz samples. Its recorded source mapping contains
one continuous range starting at zero for groups 01 and 03, and 83 separate
ranges for group 02. The app therefore has a measured reason to keep group
02's timing check unavailable while retaining independent audio and visual
checks.

## Candidate speaking intervals

Sparse single-threaded picture inspection places the first plausible
single-speaker Nairobi shot at `[11.000, 17.000)` seconds on the source
clock. Burned-in captions advance during the shot. A lower-third appears
by 14.560 seconds and a light flash at 16.520 seconds leaves the surrounding
camera composition intact. These ordinary source events remain in the
candidate.

The USGS candidate is `[1057056, 1237236)` on the 1/30000 video clock,
or `[35.235200, 41.241200)` seconds. It contains 180 pictures on the native
1,001-tick cadence. The lead speaker wears a mask, making this a genuine
occlusion control. Publisher captions cover the interval continuously.
The picture start lies between two native audio sample starts. Retain the
two clocks and record the measured offset when constructing the excerpt.
[Publisher caption file](https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/%5BSubtitleTools.com%5D%20CenterVideo_11-5.vtt)

An authored 29.97 fps codec smoke used an MP4 video track time scale of
30,000 and lossless ALAC audio. A local H.264 recompression preserved all
180 decoded picture timestamps and all 288,288 decoded 48-kHz audio samples
exactly.

The clock helper replayed decoded metadata from the original Nairobi and
USGS files. Nairobi has 150 pictures and native audio sample indices
`[528000, 816000)` on the conditional zero-origin lattice. The compatible
decoded audio origin spans 0 through 0.5 ms under closed nearest rounding.
USGS has 180 pictures and sample indices `[1691290, 1979578)` on an exact
zero-origin lattice. Both audio boundaries are 1/120000 second later than
their corresponding picture boundary.
[Clock helper](../backend/scripts/hd_excerpt_clock.py)

## Prepared genuine cases

Local preparation completed for both admitted clock intervals. Each source
produced an H.264 lossless reference and an ordinary H.264 recompression
at CRF 24, both with signed 24-bit stereo ALAC audio. The reference kept
every decoded YUV420 picture byte. The recompression changed all pictures,
and both arms preserved the exact selected audio bytes after conversion
to the common 24-bit format.

| Group | Pictures per case | Verified 48 kHz stereo samples per channel | Native video time base |
| --- | ---: | ---: | --- |
| 01 | 150 | 288,000 | 1/1000 |
| 03 | 180 | 288,288 | 1/30000 |

The first preparation attempt omitted the final video packet's duration
in the Nairobi MP4. It stored 150 packets but exposed only 149 decoded
frames. The corrected attempt sets every packet duration to the verified
native step and checks the stored packet and decoded picture clocks
independently. It retained all 330 source pictures in the references and
all 660 expected pictures across the four outputs. Source rebasing kept
the audio samples intact and moved the USGS audio start 1/120000 second
earlier relative to the video. Nairobi's original compatible origin
uncertainty remains 0 to 0.5 ms.

The prior failed attempt and the four prepared files stay on the local
archive. The completed preparation receipt has SHA-256
`89f204be932f136aee8e2c487ce17c0a14725218b675d61ddda8df4bc8487d77`.
The files are ready for the next independent serving test after the
remaining soundtrack and content-overlap checks.

To reproduce acquisition, run from the repository root and set
`CHIPMUNK_HD_SOURCE_DIR` to the chosen private holding directory. The command
downloads only the pinned original, verifies its exact publisher length and
SHA-1, and records its local SHA-256 in the terminal. It checks existing files
and uses a temporary file so an interrupted download cannot become an input.

```sh
(
  set -eu
  umask 077
  pilot_dir=${CHIPMUNK_HD_SOURCE_DIR:-backend/data/independent-evaluation/video-genuine-hd-source-admission}
  mkdir -p "$pilot_dir"
  destination="$pilot_dir/source-01.webm"
  verify_original() {
    [ -f "$1" ] && [ ! -L "$1" ] &&
      [ "$(wc -c < "$1" | tr -d '[:space:]')" = 115172735 ] &&
      [ "$(shasum -a 1 < "$1" | cut -d ' ' -f 1)" = 5b62001d250c4387edc8007d9e55fdd67c8c9d84 ]
  }
  if [ ! -e "$destination" ] && [ ! -L "$destination" ]; then
    partial=$(mktemp "$pilot_dir/source-01.XXXXXX")
    trap 'rm -f "$partial"' EXIT
    curl --fail --silent --show-error --proto '=https' --connect-timeout 15 \
      --max-time 300 --max-filesize 115172735 --output "$partial" \
      'https://upload.wikimedia.org/wikipedia/commons/f/f6/Interview_with_Jimmy_Wales_at_Wikimania_2025_Nairobi.webm'
    verify_original "$partial"
    mv -n "$partial" "$destination"
  fi
  verify_original "$destination"
  shasum -a 256 "$destination"
)
```
