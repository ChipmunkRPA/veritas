# Motion-aligned mouth detail on short neural redraws

A mouth redraw may preserve the shape and timing of speech while changing how fine skin detail persists from one picture to the next. Recent work compares temporal changes in forensic residuals and uses optical flow to account for ordinary motion. This study independently tests a small local measurement using OpenCV and the existing commercially admitted face detector. It uses no code, weights, or datasets from the cited research. [MVFNet](https://arxiv.org/html/2503.20991v1) · [GC-ConsFlow](https://arxiv.org/html/2501.13435v1) · [October 2026 motion analysis](https://arxiv.org/html/2610.11496v1) · [Face weight terms](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

## Fixed method and controls

The comparison registers each face by the eyes and measures two difference-of-Gaussian luminance bands. The finer band uses blur scales of 0.65 and 1.55 pixels on the aligned grid. The broader band uses 1.55 and 3.1 pixels. Motion between adjacent measured pictures must return to within 1.25 aligned pixels after tracking forward and backward. Each retained point must stay in the same measured region. One region covers skin around the mouth and excludes the inner mouth. The other covers nearby cheek and chin. Each correlation needs at least 48 pixels and a minimum signal variation.

For each direction the measurement compares the signed band correlations in the perioral and reference regions. A lower perioral correlation increases the response. The primary comparison subtracts the same contrast at the broader scale. It asks whether fine detail loses temporal agreement more than the broader face structures. The earlier reconstruction experiment measured absolute brightness differences after motion compensation. The present calculation standardizes each retained band separately and measures agreement in its pattern. [Earlier motion-compensated comparison](video-mouth-reconstruction-transfer-2026-11-11.md)

The existing native sample grid and shot checks bound the scan. Both adjacent edges must support a sampled center. Each arm keeps its strongest 0.40–0.88-second interval above both flanks. Each review point is the third-largest measured maximum from 56 ordinary development views, with strict exceedance. The tests use the submitted pictures alone. Exact source PCM supports replay of the selected interval. The detector uses no audio or earlier copy.

Eight GRID development recordings from four components set the points. Eight held recordings from four separate components contribute 32 neural redraws and 56 ordinary views. Ten separate CREMA-D groups contribute 20 redraws and 40 ordinary views. Eight H.264 re-encodes and four genuine-picture hard cuts bring the held total to 160. The ordinary set includes originals, conventional retiming, local lighting, resampling, blur and audio-only delay. The same exact media had earlier project exposure. The seven source-compatible GRID positives have verified visible source-time mouth motion and an unchanged soundtrack. Perceived synchrony and phoneme-level agreement remain unmeasured. [Frozen source roles and controls](video-mouth-jaw-transfer-2026-11-11.md) · [GRID publisher terms](https://zenodo.org/records/3625687) · [CREMA-D publisher terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt) · [Visible motion qualification](grid-mouth-motion-v158-result.md)

The frozen gate counts a useful interval at half or more temporal intersection over union. It requires at least four additional GRID ranges across two components, at least two source-compatible additions, and at least two additional CREMA-D ranges across two groups. It allows three additional ordinary ranges in each bank, one in the separate recompression controls, and zero in the hard-cut controls. The candidate must assess at least 75% of edited and ordinary views in each bank.

## Held outcomes

| Outcome | Direct picture comparison | Fine detail | Fine detail adjusted for broad motion |
| --- | ---: | ---: | ---: |
| Assessed views | 160/160 | 160/160 | 160/160 |
| Correct GRID redraw ranges | 1/32 | 1/32 | 3/32 |
| Misplaced GRID redraw ranges | 0/32 | 5/32 | 8/32 |
| Correct source-compatible GRID ranges | 0/7 | 0/7 | 0/7 |
| GRID ordinary flags | 0/56 | 14/56 | 11/56 |
| Correct CREMA-D redraw ranges | 1/20 | 5/20 | 3/20 |
| Misplaced CREMA-D redraw ranges | 2/20 | 3/20 | 4/20 |
| CREMA-D ordinary flags | 8/40 | 7/40 | 10/40 |
| Independent H.264 re-encode flags | 0/8 | 0/8 | 0/8 |
| Genuine-picture hard-cut flags | 0/4 | 1/4 | 2/4 |

The primary comparison found three GRID redraws in one component and three CREMA-D redraws across two groups. These six locations differ from the two direct-comparison locations. Four genuine GRID picture retimes flagged, along with ordinary resampling and brightness edits. One untouched GRID recording and its visually identical audio-delay variant both flagged. Three ordinary original CREMA-D recordings also flagged. The candidate missed all seven source-compatible redraws, exceeded the control limits, and stays outside the app.

One correct CREMA-D range spans **0.660–1.061 s** on an authored edit at **0.694–1.362 s**. The temporal intersection over union is 367/702. The native before, during and after pictures were at **0.593**, **0.760** and **1.061 s**. The aligned source audio contains 6,416 exact PCM samples over the selected range. The retained native crop locates the inspected area and its measurements do not mark edited pixels.

The held pass ran in 84.5 seconds at approximately 144.5 MiB peak reported RSS. Independent replay checked 160 file hashes, 480 decisions, 144 native crops and 25 aligned PCM intervals. For the source-compatible subset it matched seven unchanged soundtracks, 1,050 pictures and 140 exact changed-pixel masks. Identical pictures gave identical results across all eight original/audio-delay pairs. [Aggregate and source replay](evaluation-panels/video-mouth-temporal-detail-2026-11-12.json)

The retained 16-clip GRID reserve has no rendered positive counterparts. An untouched-source follow-up can qualify new native redraws and matched ordinary controls on those clips using the audited rendering and motion procedures. It also needs separate phoneme or listener agreement before assigning achieved synchrony. [Reserve inventory](grid-transfer-plan-v143-method.md) · [Native construction](grid-native-clips-v149-method.md) · [Independent mouth observer](grid-mouth-observer-v157-method.md)

## Reproduce

Use the retained mouth-jaw comparison as the prior run and a new local output directory. The final step reopens original media and keeps native crops and the corresponding WAV locally.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze development held report; do
  backend/.venv/bin/python -B -m scripts.video_mouth_temporal_detail "$phase" "$run" --prior "$prior" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_mouth_temporal_detail "$run" --prior "$prior"
```
