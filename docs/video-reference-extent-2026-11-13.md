# Estimating the changed region in a short earlier-copy video comparison

The original tile can locate a change in time while covering little of
the affected picture. This follow-up keeps the same time selector and
frozen luminance point. It estimates extent separately on an 8-pixel grid.
Each cell uses the median residual in the center half of the interval
above the larger median in the five pictures on either side. It removes
broad frame-level luminance differences, then keeps the spatial component
containing the strongest cell inside the original tile. The result is a
rectangle around that component. Low support or a component covering more
than 18% of the frame falls back to the tile. No weights were added.
[Full-reference video distortion](https://ece.uwaterloo.ca/~z70wang/publications/vssim.html)
· [Prior timing-preserving mouth test](video-paired-native-mouth-transfer-2026-11-13.md)

## Native results

On the previously consumed 17-clip diagnostic, useful time-and-pixel
locations rose from **11 to 13**. The separately timing-qualified edit
kept its exact **1.038062–1.702422 s** interval. Its middle-picture crop
covered **16,936 of 18,572 changed pixels (91.2%)**, up from 1,632 (8.8%).

The first transfer used ten distinct CREMA-D source groups from the earlier
30 fps paired study. It made fresh 720 × 480 24 fps carriers from the
retained source-index sequences with nearest-neighbor resizing. This
carrier exercise reused the authored edits and omits audio. It does not
establish synchronization on the new carriers. The baseline and larger
area both localized **20 of 20** edits. Neither flagged the 30 identical,
re-encoded and globally adjusted controls. Nine different-source
comparisons were declined. The longest source had no eligible different
match. None of ten ordinary local resamples produced a range. Both
selectors met the original 0.50 time and 0.10 spatial IoU criteria.
The result added zero new locations on that transfer, below the frozen
minimum gain of three. [Earlier CREMA-D transfer](video-reference-lossy-transfer-2026-11-11.md)

A separate scope check used six other CREMA-D source groups with exact
184 × 104 and 64 × 40 conventional edits at new positions. It kept the
frozen algorithm unchanged.

| Held outcome | Fixed tile | Connected area |
| --- | ---: | ---: |
| Large edits localized | 0/6 | 4/6 |
| Median pixel intersection over union, large | 0.085 | 0.890 |
| Small edits localized | 6/6 | 6/6 |
| Median pixel intersection over union, small | 0.607 | 0.741 |
| Identical, recompressed or global-tone controls flagged | 1/18 | 1/18 |
| Different-source pairs admitted | 0/6 | 0/6 |

The failed large cases include a timing miss and a component that exceeded
the area cap and fell back to the original tile. The larger area met the
median improvement requirement and missed the frozen five-of-six
large-edit target. The method remains offline. The experiment measures
differences between two copies and leaves single-upload detection unmeasured.

## Replay

The public ledgers include both proposed rectangles, exact mask-overlap
counts, the before/during/after frame commitments, and all assessed control
roles. They contain no source media or local paths. The standard-library
reviewer independently recomputes both frozen decisions. A second native
run reproduced every diagnostic, transfer and control measurement.

```sh
python3 -B -IS backend/scripts/review_video_reference_extent.py \
  docs/evaluation-panels/video-reference-extent-transfer-2026-11-13.json \
  docs/evaluation-panels/video-reference-extent-controls-2026-11-13.json
```

The native runner uses the previous paired-mouth run as `$previous`, plus
the archived CREMA-D lossless deliveries and the native construction
receipts. It checks the original file hashes and native truth pictures
before measuring their overlap.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze diagnostic transfer report; do
  backend/.venv/bin/python -B -m scripts.video_reference_extent_transfer "$phase" "$previous" "$run" || exit 1
done
for phase in freeze run; do
  backend/.venv/bin/python -B -m scripts.video_reference_extent_controls "$phase" "$run/result.json" "$control_run" || exit 1
done
```
