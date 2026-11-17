# Short differences between two audio tracks in one video

Some videos retain two versions of the audio. Chipmunk Verify can now locate a short difference between those tracks at the same source time. It first looks for a timing shift with the existing check. When that check supplies no location, it can look for sound that differs locally while the surrounding sound agrees in at least two frequency bands. Both excerpts and their waveforms are shown together on the video clock. The finding contributes zero weight to the AI estimate. [Existing timing comparison](video-soundtrack-pair-2026-11-14.md) · [Earlier-copy method](video-short-phrase-comparison-2026-11-17.md)

The short comparison needs two continuous mono tracks with the same start and presented length. It accepts 16 kHz PCM or qualified 48 kHz AAC-LC between 1.875 and 4 seconds. It reads only the presented audio, excluding AAC padding. The comparison can show how two streams differ. Dubbing, ordinary editing and source-track differences can explain the result, and either stream can contain the change.

## Held comparison on delivered videos

Thirty-three previously unused Lombard Grid utterances provide the video and reference track. The second track contains either a different digit from the same speaker, a same-digit replacement, an unchanged track, a local gain change, or a whole-track offset of ±200 ms. Both replacement types are positive cases because this check measures a sound change. The input staging binds all audio samples and reuses the original picture bitstream. A correct location overlaps at least 0.40 of the union with the known interval. [Publisher data and terms](https://zenodo.org/records/3736465) · [Source plan](evaluation-panels/video-word-soundtrack-second-sources-2026-11-17.json.gz)

| Case | PCM timing → with short comparison | AAC timing → with short comparison |
| --- | ---: | ---: |
| Different-digit replacement located | 0 → 26 of 33 | 0 → 26 of 33 |
| Same-digit replacement located | 0 → 24 of 33 | 0 → 24 of 33 |
| Wrong location on a replacement | 0 → 3 of 66 | 0 → 3 of 66 |
| Untouched, gain and whole-track delay controls flagged | 0 → 0 of 124 assessed | 0 → 0 of 124 assessed |
| Controls outside the short comparison scope | 8 of 132 | 8 of 132 |

The two encodings use the same sources. Two clips were shorter than the short comparison requires, leaving 31 source groups per encoding. All 396 videos were reopened with both audio streams. Their presented samples matched the pre-encode comparison, and the measured intervals and band readings reproduced. The staged audio is genuine speech from the publisher. These results measure a local difference between tracks of one file. [Comparison readings](evaluation-panels/video-soundtrack-short-difference-held-2026-11-17.json.gz) · [Delivered-video replay](evaluation-panels/video-soundtrack-short-difference-native-2026-11-17.json.gz)

On the separate GRID capture, the same comparison locates 14 of 16 different-digit replacements and 14 of 16 same-digit replacements in each audio encoding. Both methods report zero regions across 64 controls per encoding. This transfer uses the original video's aligned waveform and the two prepared audio tracks, independently of container delivery. [Original GRID recordings](https://zenodo.org/records/3625687) · [Separate capture results](evaluation-panels/video-soundtrack-short-difference-grid-2026-11-17.json.gz)

## Local app and replay

Enable `VERITAS_VIDEO_SOUNDTRACK_PAIR_REVIEW=true` and `VERITAS_VIDEO_SOUNDTRACK_SHORT_DIFFERENCE_REVIEW=true` on the local development backend. Select **Voice intervals** in the video evidence view and open **Differences between audio tracks**. A matched interval shows the same time on each track, both waveforms and playable clips. The matching sound on both sides limits the finding to that time. The Standard PDF shows both waveforms and times. JSON and the signed package retain the intervals and underlying measurements.

Five simulated $9.99 checkouts exercised the normal local worker with installed detectors. The PCM and AAC examples both report **0.900–1.340 s**. Chromium loaded both waveforms and both 0.440-second audio excerpts in each analysis. The gain-change and whole-track-delay controls produced no highlight. The same PCM input returned the same automated assessment with the comparison disabled. The queue held zero analysis jobs before each simulated payment. Signed replay matched the two original audio streams. [Prepaid app verification](evaluation-panels/video-soundtrack-short-difference-app-2026-11-17.json)

To reproduce, render the second Lombard panel with the [native whole-word replay](video-word-soundtrack-2026-11-17.md#reproduce). Use the unchanged case of the same carrier as `reference.mov`, and the chosen comparison case as `variant.mov`. This keeps the original picture and both source streams in one input.

```sh
ffmpeg -nostdin -i reference.mov -i variant.mov \
  -map 0:v:0 -map 0:a:0 -map 1:a:0 -c copy -movflags +faststart two-tracks.mov
```

The native receipt names each delivery and binds its file hash, measured time and presentation clock. Verify an individual signed export against the original with `python -m app.cli verify report.zip --public-key signer.pem --source-video two-tracks.mov --source-audio two-tracks.mov`.
