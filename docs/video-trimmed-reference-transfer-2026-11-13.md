# Compare a trimmed video against an earlier copy

**Compare with an earlier video** can now align a shorter clip with an earlier
copy after leading or trailing pictures have been trimmed. It first verifies a
single frame offset using unique, identical decoded pictures. The picture clock
must match exactly throughout the overlap. At least half of the submitted
pictures and at least 16 must support the offset. Changed ranges need unchanged,
unique neighboring frames. A comparison can return up to three changed ranges.
The existing limits of 48 MiB, 360 pictures and 12 seconds still apply.

The browser, PDF and JSON show both clip timelines and paired pictures from
before, during and after the measured change. The PDF and browser mark the area
containing changed pixels in the strongest picture. JSON records the frame and
timestamp ranges and hashes from the source files. The comparison covers the
shared pictures. The soundtrack and removed margins are outside this check.
Ordinary edits produce real picture differences, and this measurement carries
no AI confidence weight.

## Local results

Six retained Lombard sources supply exact lossless trims of the original,
same-time neural redraw, ordinary resample, and ordinary retime of mouth pixels.
Each edit spans 16 native pictures. Their sources and native clocks were
verified before this test. This trial strips audio from the trimmed test files.
Lossy re-encodes, another source, and nonmatching native clocks are controls.
One of the six neural redraws was independently verified to preserve the local
speech event in its untrimmed source delivery. [Source and terms](lombard-fresh-musetalk-2026-11-13.md)

| Held deliveries | Current equal-clock comparison | With trim alignment |
| --- | ---: | ---: |
| Same-time neural redraw localized | 0/6 | 6/6 |
| Ordinary visual edits localized | 0/12 | 12/12 |
| Unchanged trims assessed with zero differences | 0/6 | 6/6 |
| Wrong-source, lossy or changed-clock pairs admitted | 0/18 | 0/18 |

All 18 reported edits matched the authored start and exclusive end on both
clips. Two initial encoder attempts did not preserve the full trim. The final
run disabled the MOV edit list and separately verified the actual decoded
frame count and pixel hashes before scoring. A separate native replay rebuilt
all 42 deliveries. The authenticated API test verified the PDF, JSON, source
frame hashes and rejection of a changed overlap in the JSON export. The
browser component test checked that both timestamps appear separately. The
application comparator reproduced all 42 outcomes and produced 54 pairs of
source previews. [Comparison receipt](evaluation-panels/video-trimmed-reference-transfer-2026-11-13.json)

## Replay

To reproduce the study, use the retained native Lombard and MuseTalk source
directory from the source qualification study and an empty scratch path.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for stage in freeze run review; do
  backend/.venv/bin/python -B -m scripts.video_trimmed_reference_transfer "$stage" "$source" "$scratch" || exit 1
done
```

To verify a downloaded JSON comparison against its two original files, use
the same FFmpeg decode environment as the app.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.compare_video_references \
  "$earlier" "$submitted" "$export" --verify
```
