# Video soundtrack dropouts at 44.1 kHz

Chipmunk Verify's local Full video analysis now checks for a near-silent
soundtrack while a visible mouth keeps moving in eligible clips with a
continuous mono 44.1 kHz AAC-LC track. The analysis links the time range
to the waveform, a playable excerpt and original frames with mouth crops
before, during and after. The standard PDF and JSON use the same ranges.
Muting, recording dropouts and quiet mouth movement can all produce
this observation. It carries zero weight in the overall AI assessment.

The decoder verifies the original 44.1 kHz clock before resampling.
The clip must start at zero and have one supported audio stream. The
resampled clock must also remain continuous. For a source extent of
`n` native samples, the review uses `floor(n × 16000 / 44100)` complete
analysis samples, omitting AAC padding and any final fraction of an
analysis sample. The picture comparison keeps the exact decoded
presentation timestamps. Scope is up to five shots and 15 seconds of
presented sound. The source and audio extent are replayable from the
original upload. This extends the per-shot near-silence
measurement. The spoken-sound timing comparisons have their own
separate format scope. [Earlier 29.97 and 30 fps check](video-sound-dropouts-30fps-2026-11-17.md)

## Paired local test

The 392 new deliveries use the same recorded mouths, untouched controls
and labeled audio changes as the earlier PCM and 48 kHz runs. The replay
checks each parent PCM delivery by hash, encodes its audio as 44.1 kHz
AAC-LC at 112 kb/s, and verifies the unchanged encoded picture stream
and every presentation timestamp. The 32 source actors are the same
people as in the previous panel. This comparison measures transfer to a
different audio encoding. Correct locations have at least 0.40
temporal intersection over union with the known interval.
[CREMA-D dataset terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt)

| Picture presentation | Correct hard mutes | Correct faded mutes | Matched controls flagged |
| --- | --- | --- | --- |
| Native 29.97 fps, one shot | 30/32 | 30/32 | 0/96 |
| Exact 30 fps, one shot | 30/32 | 30/32 | 0/96 |
| Exact 30 fps, three shots | 11/12 | 11/12 | 0/48 |

The previous local gate skipped the 44.1 kHz encoding. The extended
route assessed all 392 deliveries and reproduced the 48 kHz check's
case-level locations on 142 of 152 staged mutes. Both carriers missed
the other ten. The findings reached the overlap rule in every alerted
edit. All 240 unchanged, gain, partial level reduction and cross-cut
controls had zero qualifying findings. The title-card shots in the
montages stayed outside the checked scope.
[Local paired result](evaluation-panels/video-sound-dropouts-aac441-2026-11-17.json.gz)

## Checkout and export

The local app checked one native video and one montage, alongside
their unaltered controls and a replay with the option disabled. The
six USD 9.99 payments were simulated. Analysis jobs queued after
payment confirmation. The browser, PDF and JSON agreed on the
1.325–1.675 s and 4.570–5.085 s ranges. Both signed packages checked
against the submitted originals. Each browser example decoded three
original frames, three mouth crops, the waveform and the audio excerpt.
Turning the option on preserved the same overall AI assessment in
both paired app replays.
[Checkout and export receipt](evaluation-panels/video-sound-dropouts-aac441-app-2026-11-17.json)

## Reproduce

Restore the exact PCM deliveries first using the
[prior source recipe](video-sound-dropouts-30fps-2026-11-17.md#reproduce-the-checks).
The same FFmpeg 7.0.2 installation produced the retained and new
deliveries. Run the following from the repository root.

```sh
repo=$(pwd)
export PYTHONPATH="$repo/backend"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
assets="$repo/models/research/video-lip-timing"
reference="$repo/docs/evaluation-panels/video-sound-dropouts-30fps-2026-11-17.json.gz"
module=scripts.video_silent_mouth_aac441_panel
runner="$repo/backend/.venv/bin/python"
python3 -B -S -m "$module" review \
  "$repo/docs/evaluation-panels/video-sound-dropouts-aac441-2026-11-17.json.gz" "$reference"
"$runner" -B -m "$module" measure "$previous_panel" "$reference" "$assets" "$run/deliveries"
npm --prefix frontend run build
for clock in native-29.97 montage-30; do
  (cd backend && "$runner" -B -m scripts.video_lip_multishot_app \
    "$run/deliveries" "$assets" "$run/app-$clock" --carrier aac441 --clock "$clock" --browser)
done
```

The public result contains delivery, source and picture-scope commitments
and the checked intervals. Recorded people, clips and crops stay in the
local test directories.
