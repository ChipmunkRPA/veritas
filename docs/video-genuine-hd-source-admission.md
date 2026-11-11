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

## First acquisition

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
This prefix check leaves the selected speaking interval and its sample
mapping to be verified. Groups 02 and 03 remain prospective.

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
