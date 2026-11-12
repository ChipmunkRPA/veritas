# Earlier H.264 copies in the paired video comparison

At the existing operating point, independently encoding the earlier copy as H.264 preserved all 44 useful redraw locations from the earlier lossless-reference trial. The GRID group met the frozen criteria, and CREMA-D exceeded the allowed number of ordinary delivery flags. The joint trial failed the release gate. [Aggregate and independent replay](evaluation-panels/video-reference-h264-earlier-transfer-2026-11-11.json)

## Fixed test

The trial reuses the 184 exact submitted files, authorized recordings, source clocks, and authored intervals from the lossless-reference study. It makes a separate H.264 4:2:0 earlier copy for each of the 18 source recordings with a medium encoder preset, CRF 23 and a fixed 48-picture GOP. The previously retained submissions used a separate veryfast encode at CRF 28 or CRF 33. The benchmark also creates 36 tandem encodes, two from each new earlier copy, and nine unrelated CREMA-D submissions whose native clock prefixes match the target. The unrelated control selects the shortest eligible recording from a different known source group and truncates only excess pictures at the end. [Source and earlier comparison](video-reference-lossy-transfer-2026-11-11.md)

The running comparison keeps the existing tile measurement and point. Both clips require equal native relative picture timestamps, dimensions, and orientation. No image or time alignment is estimated. The test holds the original shared-picture requirements, paired luminance normalization and flank comparison fixed. A useful location requires at least 0.50 frame-interval intersection over union and the existing minimum changed-pixel fractions in the selected native tile. Independent replay also checks that the exact native clock gives the same half-overlap decision. Each bank needed to retain at least 90% of the prior neural locations, reach three source groups, assess at least 80% of positives and neutral/tandem controls, add no unrelated or tandem flags, and flag at most one neutral delivery.

## Results

| Held outcome | GRID at 360 × 288 and 25 fps | CREMA-D at 480 × 360 near 30 fps |
| --- | ---: | ---: |
| Useful neural redraw locations | 24/32 | 20/20 |
| Prior useful neural locations retained | 24/24 | 20/20 |
| Useful ordinary local edit locations | 16/40 | 20/30 |
| Neutral delivery flags | 0/24 | 2/30 |
| Tandem re-encoding flags | 0/16 | 1/20 |
| Unrelated-reference flags | 0/8 | 0/9 |

All edited and ordinary delivery views were assessed. All 17 unrelated references lacked enough shared picture content. CREMA-D flagged one independent CRF 33 delivery, one global-tone delivery, and one tandem CRF 33 delivery. The trial retained the same 36 useful authored ordinary local edit locations. The seven source-compatible visible-motion GRID redraws remained misses. The images and clips had previous project exposure.

The one-thread run took 158 seconds and reached about 254 MiB reported peak RSS. Independent replay checked 18 earlier clocks, 229 paired deliveries, 546 selected native pictures and crop bounds, and 88 actual authored changed-pixel counts. Per-case media and source identifiers stay local.

## Reproduce

Use the existing private paired video study directory as the prior run and a fresh private output directory. The earlier and submitted H.264 files retain their separately observed timestamps and picture dimensions.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze held report; do
  backend/.venv/bin/python -B -m scripts.video_reference_h264_earlier_transfer "$phase" "$run" --prior "$prior" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_reference_h264_earlier_transfer "$run" --prior "$prior"
```
