# Brief soundtrack differences between two video copies

The local earlier-video comparison now checks for a brief audio change
between intervals where the two soundtracks agree. Use a short H.264 pair
with matching decoded pictures, a common picture clock near 24 fps, and
equal mono audio presentations starting on the first picture. The tested
size is 720 × 480 and the duration is about two to four seconds. The
comparison accepts 16 kHz PCM and qualified 48 kHz AAC-LC. Audio padding
past the last presented sample is excluded. [Existing native clock check](video-short-native-soundtrack-2026-11-16.md)

The check keeps a qualifying range from the earlier version of the app and
gives the timing comparison priority when it finds a local move. Otherwise
it uses 60 ms windows spaced 20 ms apart to examine candidate differences
from 160 to 440 ms. The sound must agree before and after the interval in
at least two frequency bands, with much lower agreement inside. The report
shows the shared time, the frames before, during and after in both copies,
the playable excerpts and waveforms, and the measured bands. The cause of
the difference remains unresolved and carries zero weight in the AI estimate.

## Local results

The test used 24 previously unused Lombard Grid utterances from 24 speakers.
The known changes replace one digit with a rendition by the same speaker
within the same recording condition. A second track replaces that digit with
the same digit in another rendition. The controls keep the original sound,
change its gain in the selected interval, or shift the whole soundtrack by
200 ms. Each result needs at least 0.40 intersection over union with the
known audio interval to count as a correct location. [Source data and CC BY 4.0 terms](https://zenodo.org/records/3736465)

| Fresh test | PCM to PCM | PCM to AAC | AAC to AAC |
| --- | ---: | ---: | ---: |
| Different-digit edit located | 12 → 19 of 24 | 12 → 19 of 24 | 12 → 19 of 24 |
| Same-digit edit located | 12 → 19 of 24 | 12 → 19 of 24 | 12 → 19 of 24 |
| Added misplaced range on each edit type | 1 of 24 | 1 of 24 | 1 of 24 |
| Added flags on unchanged, gain, and whole-track delay controls | 0 of 96 | 0 of 96 | 0 of 96 |

The three pairings reuse the same recordings. The earlier seven-clip panel
improved from 3 to 5 correct different-word locations in each pairing. The
earlier 36-clip panel improved from 24 to 26 in each pairing and includes
two unassessed sources. No new control alerts appeared on either panel. In
each fresh pairing the check leaves 48 whole-track delays and two of 48
unchanged or gain-only controls unassessed. Two substitutions of each type
are also unassessed in each pairing. The test scope covers staged human
speech substitutions and exact picture pairs.
Generated voices, unscripted footage and re-encoded pictures require separate
testing. The [single-upload check](video-mouth-patch-speech-2026-11-17.md)
keeps its existing model because neither new mouth model variant added a
correct location on the fresh panel. [Per-pair measurements](evaluation-panels/video-short-phrase-benchmark-2026-11-17.json.gz)

## Use locally

Enable `VERITAS_VIDEO_REFERENCE_SHORT_PHRASE_DIFFERENCE=true` along with
`VERITAS_VIDEO_REFERENCE_COMPARISON_ENABLED`, `VERITAS_VIDEO_REFERENCE_SOUND_DIFFERENCE`,
and `VERITAS_VIDEO_REFERENCE_SHORT_NATIVE_SOUND_DIFFERENCE` in development.
Open a completed video analysis, choose **Compare with an earlier video**,
and select **Include experimental soundtrack comparison**. Turning off the
new option restores the previous native soundtrack comparison. The comparison
downloads have the same interval and source evidence as the browser.
The local playback test shows a difference from 1.460 to 1.900 seconds in
both copies. Each excerpt lasts 0.440 seconds. The six pictures embedded in
the PDF match the decoded source pixels recorded in JSON.

## Replay

Use the exact publisher archives in the [source preparation instructions](video-neural-phone-clock-2026-11-14.md#independent-replay).
The measurements have digests for every rendered audio track. Reconstruct
the fresh videos with the [playable export command](video-mouth-patch-speech-2026-11-17.md#reproduce).
The native replay reads the delivered videos through the same core comparison
as the browser and checks the returned ranges against the measurement packet.
The paid replay uses simulated USD 9.99 checkouts in isolation and verifies
the legacy and current JSON downloads against both inputs. It checks the
localized evidence in the browser and PDF. [Native comparison receipt](evaluation-panels/video-short-phrase-native-2026-11-17.json.gz)
· [Local app receipt](evaluation-panels/video-short-phrase-app-2026-11-17.json)

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
panels=docs/evaluation-panels
measured="$panels/video-short-phrase-benchmark-2026-11-17.json.gz"
python3 -B -S -m scripts.video_short_phrase_comparison verify "$measured"
"$runner" -B -m scripts.video_short_phrase_comparison measure "$sources" "$run/signal.json.gz" \
  "$panels/video-mouth-patch-sources-2026-11-17.json.gz" "$panels/video-mouth-patch-fresh-sources-2026-11-17.json.gz"
"$runner" -B -m scripts.video_short_phrase_comparison native "$run/visual-fresh" "$measured" "$run/native.json.gz"
npm --prefix frontend run build
"$runner" -B -m scripts.video_short_native_soundtrack_app "$run/visual-fresh/clips" "$run/short-app" --browser --short-receipt "$measured"
```
