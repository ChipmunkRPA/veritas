# Comparing brief soundtrack changes after video re-encoding

A local comparison of two copies can find a short replacement in the soundtrack
when the pictures still correspond after a separate encode. This experiment
checks short, upright 720 × 480 H.264 clips with equal native picture timestamps
near 24 frames per second. The earlier copy must be available. The result
measures a difference between recordings and carries zero weight in the
standalone AI assessment. [Existing exact-picture comparison](video-short-phrase-comparison-2026-11-17.md)

## Method and held sources

The comparison keeps the earlier soundtrack readings and their operating
points. It checks both 80 × 60 and 160 × 90 picture grids at every native
timestamp, using the existing whole-picture and five-picture error limits.
The mean luma error of every 5 × 5 grid cell must also stay at or below 13
in each individual picture. This last check catches short disagreements that
could be diluted by the five-picture median. Correspondence is restricted to
the same picture clock and both supplied native frame sequences.

The holdout uses 14 Lombard Grid speakers absent from the previous paired
short-phrase benchmark. Each contributes a new target recording and two new
matched recordings of the same speaker. We excluded known target and donor
recordings and file digests from the public source rosters. All 54 speakers
from this dataset had appeared in other project work. Twenty-one were outside
the previous paired benchmark, and 14 supplied new recordings with the
required clock and audio alignments. These 14 people had previously supplied
data to a separate visual learner. This comparison uses only measured pixels
and audio samples, with zero learned weights. [Lombard Grid publisher and CC BY 4.0 terms](https://zenodo.org/records/3736465)

Each target supplies a same-digit and a different-digit soundtrack
replacement. The known intervals are 320–450 ms. A match must have at
least 0.40 intersection over union with the edited interval. Controls leave
the sound unchanged, change local gain or shift the full soundtrack by
200 ms in either direction. Two separately encoded picture controls introduce
a two-frame and a ten-frame colored rectangle respectively. These controls
check whether visually discordant clips are admitted even if an acoustic
difference could be found. They do not establish sensitivity to subtle mouth
or face edits.

The reference pictures use x264 CRF 23. The two same-picture deliveries
encode the publisher frames independently at CRF 28 and CRF 34. The frozen
acceptance requires at least 14 source groups, 85% picture and audio coverage,
80% of the exact-picture localization count, useful localization in eight
groups, at most one new ordinary alert group, and zero admitted picture
controls. [Frozen inputs and decisions](evaluation-panels/video-short-phrase-reencode-plan-2026-11-18.json.gz)

## Local results

| Picture delivery | Admitted picture pairs | Different-digit locations | Same-digit locations | Ordinary audio alerts |
| --- | ---: | ---: | ---: | ---: |
| Exact-picture benchmark on the same audio | 14/14 | 14/14 | 13/14 | 0/56 |
| Independent CRF 28 encode | 14/14 | 14/14 | 13/14 | 0/56 |
| Independent CRF 34 encode | 3/14 | 3/14 | 2/14 | 0/56 |

Each row gives the count **per audio carrier pairing**. PCM-to-PCM,
PCM-to-AAC-LC and AAC-LC-to-AAC-LC returned the same numbers on the
same sources. The exact-picture acoustic benchmark located 8/14 in each
edit category with the earlier check, then located another 6 different-digit
and 5 same-digit replacements with the brief-phrase fallback. The moderate
encode preserved every location. Every reported range overlapped its staged
edit. The guard rejected all 28 authored picture modifications. The CRF 34
arm failed the frozen coverage and location criteria.
The quality values describe two tested encodes. The comparison makes its
decisions from measured picture differences. Only 12 of the 56 ordinary
audio controls per carrier reached the acoustic check at CRF 34.
[Held measurements and counts](evaluation-panels/video-short-phrase-reencode-held-2026-11-18.json.gz)

The current exact-picture application replay reported **0.980–1.340 s**
on the identical-picture copy. It withheld the same audio replacement when
all 53 pictures differed through independent encoding. The prospective
comparison verified both copies' original frames at native indices **22,
27 and 32**. The two copies had the same measured times and distinct RGB
digests at each of those positions. All six exported PNGs decoded back
to their respective source pixels. These results cover paired recordings
of staged short speech edits. Standalone lip sync and face swaps, source
trimming, synthetic voices and unscripted clips require separate evaluation.
[Native application replay](evaluation-panels/video-short-phrase-reencode-native-2026-11-18.json)

## Local option

Enable `VERITAS_VIDEO_REFERENCE_SHORT_PHRASE_REENCODE=true` with the
[existing short-phrase settings](video-short-phrase-comparison-2026-11-17.md#use-locally).
Then choose **Compare with an earlier video** and **Include experimental
soundtrack comparison** from a completed analysis. This re-encoding
extension admits 53–72 native pictures at the same verified timestamps.
The ordinary view and PDF show both earlier and analyzed frames before,
during and after the reported interval. The ordinary view and JSON include
both playable excerpts. Use `scripts.compare_video_references --verify` with
both supplied originals to recompute the picture and audio evidence in the
downloaded JSON. Set the option to `false` to restore exact-picture admission.

## Reproduce

Use the publisher archives from the [source preparation instructions](video-neural-phone-clock-2026-11-14.md#independent-replay).
The implementation uses the installed FFmpeg 7.0.2 and keeps original media,
rendered encodes and waveforms in local scratch. The receipts keep source
references, coordinates, hashes, measured differences and decisions.
The native measurement uses the pinned application dependencies from the
[baseline revision](https://github.com/ChipmunkRPA/veritas/commit/b550a9a4db5db5f437bf13daa6e038d31ae012ca).
For native measurement, copy the released evaluation runner into a separate
checkout of that baseline so its imports use the pinned dependencies. The
numeric replay needs only the released runner and receipts.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
panels=docs/evaluation-panels
plan="$panels/video-short-phrase-reencode-plan-2026-11-18.json.gz"
held="$panels/video-short-phrase-reencode-held-2026-11-18.json.gz"
python3 -B -S -m scripts.video_short_phrase_reencode verify "$plan" "$held"
backend/.venv/bin/python -B -m scripts.video_short_phrase_reencode \
  measure "$sources" "$plan" "$run/video-short-phrase-reencode.json.gz"
```
