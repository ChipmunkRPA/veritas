# Earlier-copy soundtrack timing in widescreen video

Chipmunk Verify can compare short 720p copies exported with different
H.264 compression. It looks for the same sound at a new time after
checking that pictures still correspond. A reported shift includes
both sound ranges, waveforms, playable excerpts, and the original
pictures before, during, and after. The result compares two files.
It carries zero AI-authorship weight.
[Earlier 480×360 comparison](video-soundtrack-reencode-2026-11-13.md)

## Frozen comparison

Each pair has 120 upright 1280×720 pictures on an exact 30 fps
relative presentation clock, H.264 8-bit 4:2:0 pictures and continuous
16 kHz decoded sound starting at the first picture. The videos come
from 14 publisher-verified source files. The excerpts were adapted to
this delivery profile, so the results do not estimate performance on
native camera 720p. Eleven source recordings were held for this
comparison. These recordings had prior project exposure and this study
used different offsets from the earlier short timing and adjacent-repeat
studies. The source plans record the publisher commitments and licenses.
[Conversation source plan](evaluation-panels/video-mouth-hd-hold-2026-11-12.json)
· [Additional public sources](evaluation-panels/video-lipsync-repeat-public-speech-sources-2026-11-13.json)

Compare the existing 80×60 whole-picture and local grid with a 160×90
grid, separately and together. The same five-frame local statistic
and the existing limits apply to both. At each grid, the largest
five-frame median across 5×5 cells must be at most 6.27 luma levels.
Frame mean differences must stay within the existing 65th- and 90th-
percentile limits. The waveform check uses the existing 160 ms windows,
40 ms hop, signed search and requirement for same-time sound outside
the candidate interval. New research distinguishes timing from the
content correspondence needed to interpret it. This check uses neither
external research code nor audiovisual model weights. The existing
face-geometry weights were used to build mouth-only controls.
[Timing and content evaluation](https://arxiv.org/abs/2607.00726)
· [Face-geometry license](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

Each source supplied two 200 ms signed moves inside a 900 ms segment.
Controls include a new encode alone, gain, filtering, a constant
sound offset, a picture-only move, a whole-picture or mouth-only move
together with the soundtrack, and independent speech. A useful result
needs a signed error of at most 16 samples and 0.60 temporal intersection
over union with the authored interval. PNG witnesses and playable WAVs
reuse the original decoded pixels and samples from the two files.

## Results

Both PCM and AAC produced the same outcome. Counts shown are for each
carrier. The carriers are repeat measurements of the same source and edit.

| Picture requirement | Edited pairs qualified | Useful shifts | Control flags |
| --- | ---: | ---: | ---: |
| 80×60 grid | 20/22 | 17/22 | 1/88 |
| 160×90 grid | 20/22 | 17/22 | 0/88 |
| Both grids | 20/22 | 17/22 | 0/88 |

One mouth-only synchronized control had a 4.96 coarse-grid difference
and a 7.92 fine-grid difference. Requiring both rejected it. One source
lacked an observable mouth at all required control frames, so its
synchronized mouth edit stayed in the planned denominator and was
unavailable in each carrier. The other five missed edits comprised two
picture-gate exclusions and three waveform misses.
[Numeric receipt](evaluation-panels/video-soundtrack-hd-transfer-2026-11-14.json)

## Local use and replay

In a video analysis, open **Compare with an earlier video** and supply
the previous copy. The re-encode check is available when
`VERITAS_VIDEO_REFERENCE_SOUNDTRACK_REENCODE=true`. The normal timing
check for a single clip remains separate. The browser and PDF display
the 720p profile and both source positions. JSON retains both picture
grids, the signed movement, the actual sample ranges, and six original
frame commitments. The app labels the finding as timing relative to the
supplied copy. It makes no determination about AI or original lip sync.

The local API test verifies a +200 ms result through PDF and exported
JSON replay. A separate check of the actual held AAC files reproduced
the **1.520–2.400 s** interval matching **1.320–2.200 s** in the earlier
copy, and verified that the synchronized mouth control yields no
soundtrack range or playback assets.

To recreate the comparison, collect the verified WebM sources from the
two public plans into one folder. The runner checks their hashes. The
repeat run reproduced all case records and aggregate outcomes.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze score review; do
  backend/.venv/bin/python -B -m scripts.video_soundtrack_hd_transfer \
    "$phase" "$sources" "$new_output" || exit 1
done
```

To verify the finished comparison against the two original files, use
`python -m scripts.compare_video_references earlier.mp4 analyzed.mp4 export.json --verify`
from `backend`. Older 480×360 comparisons can still be replayed with
the recorded method.
