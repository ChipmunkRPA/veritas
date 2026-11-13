# Repeat-speech timing on fresh actors and conversations

The paired speech-and-mouth check added four useful locations in each
carrier on expressive recordings from different actors. It added none in
the conversational transfer. The transfer also produced one observation
on a synchronized mouth-and-sound copy in each carrier. The check stays
experimental. [Expressive actor receipt](evaluation-panels/video-lipsync-repeat-fresh-actors-2026-11-13.json) ·
[Conversational receipt](evaluation-panels/video-lipsync-repeat-public-speech-2026-11-13.json)

## Fixed method and sources

Both tests reused the earlier acoustic nomination, static mouth guard,
and signed mouth-motion check with surrounding picture context. They kept
the 0.65 motion point and the earlier rule for advancement. A useful result
must have at least 0.50 temporal intersection over union with **both** known
witness ranges. The combined check must add at least six useful locations
across four groups, retain all static locations, add at most one ordinary
flag and none on synchronized or global-shift controls, and assess at least
75% of planned deliveries. The output locates two related speech segments
and the accompanying picture measurements. It gives no direction of change
or AI estimate. [Frozen comparison](video-lipsync-repeat-motion-2026-11-13.md)

The first roster excludes all 31 actor codes listed in the earlier repeat
experiment, including its source-duration declines. The pinned CREMA-D
publisher tree supplied four distinct sentence choices per selected actor.
Only happy or sad takes with the publisher's `XX` intensity code were
eligible. The fixed hash order selected 16 actors and 64 source files.
Publisher Git LFS pointer hashes and downloaded SHA-256 values matched.
Fourteen actors supplied the required 2.52 seconds of picture and sound.
The other two remain in the planned denominator. The published project
has used other CREMA-D recordings with these actors. Actor separation
applies to this repeat-method sequence. [CREMA-D pinned terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt)

The second test used four people in separately published Wikitongues and
NASA recordings. The source plan fixes the original URLs, terms, source
hashes and three candidate offsets per recording. For each, the first
2.52-second excerpt with exactly one measurable face at the three fixed
geometry points and at least 45% speech was used. The excerpts were adapted
to 480 × 360, 25 fps, and 16 kHz mono audio on the delivery grid. The four
people had no known overlap with the previous actor panels. Source clips
and adaptations stay local. [Pinned sources and attribution](evaluation-panels/video-lipsync-repeat-public-speech-sources-2026-11-13.json)

Each qualified source supplies four known timing edits and nine controls.
The controls include unchanged delivery, local gain and low-pass changes,
a whole-track shift, picture copy, synchronized full-picture and soundtrack
copy, synchronized mouth and soundtrack copy, an independent actor's
speech, and a quiet repeated bed. PCM and AAC are matched deliveries of
those same cases and do not represent different speakers or independent
observations. The study measures authored soundtrack operations. Neural
mouth redraws with unchanged speech timing remain a separate question.

## Held results

| Bank and carrier | Static useful | Picture-context useful | Union useful | Ordinary flags in union |
| --- | ---: | ---: | ---: | ---: |
| Expressive actors, PCM | 16/56 | 19/56 | 20/56 | 0/126 |
| Expressive actors, AAC | 15/56 | 18/56 | 19/56 | 0/126 |
| Conversations, PCM | 9/16 | 8/16 | 9/16 | 1/36 |
| Conversations, AAC | 10/16 | 8/16 | 10/16 | 1/36 |

The expressive clips added four useful locations across four actors in
each carrier, with no losses and no added ordinary observations. The
measured count was 172 of 208 planned deliveries per carrier. The frozen
minimum gain of six was unmet. One exact recovery in both carriers used
**1.440–1.760 s** and **1.760–2.080 s**, with measured source frame indices
36–43 and 44–51. Independent FFprobe and FFmpeg replay verified 2,331
picture timestamps, 1,503,360 samples, and 678 native mouth crops from
selected expressive deliveries.

Every conversational delivery was measured. The same known synchronized
mouth-and-sound copy was flagged by both arms in both carriers. Its selected
span slightly extended across the authored boundaries. The signed motion
distance remained about 0.967. This case shows that a high motion response
can still occur on a synchronized copy. The combined check added zero
useful conversational locations and does not replace the existing app
check. The app's existing timing arm produced no measured windows on
these 2.52-second deliveries.

## Replay

Use the [fixed expressive source plan](evaluation-panels/video-lipsync-repeat-fresh-actors-sources-2026-11-13.json)
to replay the actor panel. The acquisition verifies all 64 publisher files
before the measurement freeze. To rebuild the plan itself, run `intake`
with `--tree` pointing to the pinned CREMA-D recursive Git tree JSON and
`--earlier` pointing to the prior picture-context `freeze.json`. It checks
that the earlier roster matches its published digest.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir "$run"
cp docs/evaluation-panels/video-lipsync-repeat-fresh-actors-sources-2026-11-13.json "$run/intake-freeze.json"
for phase in acquire freeze score; do
  backend/.venv/bin/python -B -m scripts.video_lipsync_repeat_fresh_actors "$phase" "$run" || exit 1
done
```

For the conversational test, download the four original `url` values from
the source plan into a fresh directory as `<id>.webm`. Both phases check
their pinned hashes. The runner also verifies generated video clocks and
selected native mouth crops independently from the prediction process.

```sh
for phase in freeze score; do
  backend/.venv/bin/python -B -m scripts.video_lipsync_repeat_public_speech "$phase" "$speech_run" || exit 1
done
```

The two standard-library readers recompute the planned inputs, useful
ranges, held control counts and locked decision. Both matched on Python
3.11 and 3.12. No raw media, facial features or source-local paths are
included in either outcome ledger.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_video_lipsync_repeat_fresh_actors \
  docs/evaluation-panels/video-lipsync-repeat-fresh-actors-2026-11-13.json
PYTHONPATH=backend python3 -B -S -m scripts.review_video_lipsync_repeat_public_speech \
  docs/evaluation-panels/video-lipsync-repeat-public-speech-2026-11-13.json
```
