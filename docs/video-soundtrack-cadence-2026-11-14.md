# Earlier-copy soundtrack timing at 24 and 25 fps

The comparison looks for a local soundtrack movement between two exports
whose pictures still correspond. The existing 720p check covers an exact
30 fps clock. This transfer tests 24 and 25 fps on the same four-second
duration with both picture grids and the same waveform operating points.
[30 fps evaluation](video-soundtrack-hd-transfer-2026-11-14.md)

## Frozen comparison

Fourteen publisher-verified recordings supplied different excerpts from the
30 fps study. The speakers had already appeared in prior local studies.
The test prepared 1280 × 720 H.264 8-bit 4:2:0 copies with 96 or 100
verified picture timestamps on their exact relative clocks. Audio is
decoded to 16 kHz mono after delivery as lossless PCM or 64 kbps AAC.
The carriers and frame rates are repeated deliveries from the same
recordings. The comparison keeps both 80 × 60 and 160 × 90 picture grids,
including the existing limit on the largest five-frame median change.
The waveform comparison keeps its 160 ms window, 40 ms hop, and required
same-time flanks. The app comparison requires no model weights.
The published face geometry asset was used to construct the mouth controls.
[Conversation source plan and licenses](evaluation-panels/video-mouth-hd-hold-2026-11-12.json)
· [Additional source plan and licenses](evaluation-panels/video-lipsync-repeat-public-speech-sources-2026-11-13.json)
· [Face-geometry asset license](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

Each pairing supplied a 200 ms sound shift in either direction inside a
960 ms interval. Controls preserve the sound, change its gain or bandwidth,
move it by a constant offset, substitute separate speech, move only the
pictures, or move sound with all or part of the picture. Synchronized
controls move five pictures, approximately 208.3 ms for 24 fps and
exactly 200 ms for 25 fps. The sound intervention uses the nearest
whole sample. If the mouth geometry needed to construct one of the controls
is unavailable, that control remains in the planned denominator.

A result counts if it locates the edited interval at **0.60 temporal IoU**
or more and measures the signed movement within **16 samples**. Each
cadence must find at least half the planned edits and flag zero controls
in both carriers to qualify for the local app.

An initial run reused the reference filename for the earlier-shift
delivery. That collision invalidated the first run. The corrected runner
uses distinct reference and delivery filenames and checks that the
untouched control exactly matches the fixed reference PCM, while both
edited deliveries differ from it. The operating points and source roster
were retained.

## Results and local use

| Frame rate | Sound delivery | Useful locations /28 | Picture-qualified edits /28 | Control flags /112 |
| --- | --- | ---: | ---: | ---: |
| 24 fps | PCM | 21 | 26 | 0 |
| 24 fps | AAC | 21 | 26 | 0 |
| 25 fps | PCM | 21 | 26 | 0 |
| 25 fps | AAC | 21 | 26 | 0 |

The same 21 shifts were located in all four delivery sets, including
10 moves earlier and 11 later. Seven edits were missed per set.
Picture correspondence excluded both edits from one heavily recompressed
source, and the waveform check missed five more edits. Both cadences
qualified. The app now accepts four-second 720p pairs at 24, 25, and
30 fps through **Compare with an earlier video** when
`VERITAS_VIDEO_REFERENCE_SOUNDTRACK_REENCODE=true`. The browser and
exports identify the measured cadence. The result measures timing
relative to the supplied copy and contributes zero AI confidence.

The local checkout check used simulated USD 9.99 payment confirmation
before analysis. The same source's +200 ms edit was reported at
1.600–2.560 s in the analyzed clip and 1.400–2.360 s in the earlier clip.
The two PDFs each show six corresponding pictures and both waveforms.
The JSON replay regenerated the six source positions and two sound
excerpts. In separate comparisons on the same correctly prepared
source, the −200 ms edit was found at both rates, the two synchronized
controls failed picture correspondence, and a constant shift produced
no local timing finding.

## Replay

The [scalar receipt](evaluation-panels/video-soundtrack-cadence-2026-11-14.json)
records both picture-grid readings, sample commitments, and selected
waveform windows. The reviewer checks the full roster, the paired evidence
coordinates, and the adoption decision with standard Python.

For native replay, collect the originals from the two source plans, named
`<name>.webm` or `<id>.webm`. The runner accepts multiple source directories
and checks the publisher file hashes before extraction.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
receipt=docs/evaluation-panels/video-soundtrack-cadence-2026-11-14.json
python3 -B -S -m scripts.review_video_soundtrack_cadence "$receipt"
for phase in freeze score; do
  backend/.venv/bin/python -B -m scripts.video_soundtrack_cadence "$phase" \
    "$new_run" "$conversation_cache" "$other_public_cache" || exit 1
done
cmp "$receipt" "$new_run/receipt.json"
```
