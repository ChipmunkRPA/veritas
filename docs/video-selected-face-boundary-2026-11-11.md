# Face-boundary residuals in selected video intervals

The existing video checks compare face appearance over time and noise levels inside and around the face. This experiment asks whether the frames in an already selected interval show a repeatable difference across one cheek or jaw boundary. It keeps the original flag and endpoints and evaluates a separate cue for positive corroboration. Unavailable measurements carry no inference about the file. [Existing frame noise check](../backend/app/detectors/video/adapters/temporal.py) · [Original selected intervals](video-selected-landmark-reliability-2026-11-11.md)

## Comparison

The replay uses the previous six RTFS source groups. Their 36 known swaps cover two swap families and three edit lengths per source. Their 60 matched views cover unchanged video and three ordinary changes. These clips already informed the landmark reliability study. Each of the three v202 deliveries shares one genuine camera recording whose existing appearance check selected the same false interval. The three prepared v290 deliveries had no eligible appearance window and remain a separate coverage case. [Native interval construction](video-native-inserts-v95-method.md) · [Saved camera results](evaluation-panels/video-vcd2-full-app-eight-triplets-v202.json) · [Prepared camera controls](video-vcd2-matched-controls-v290-result.md)

The runner recovers the exact centers of each incumbent flag from the committed native frame hashes. It independently verifies source bytes, native RGB pictures and presentation timestamps. The cue examines five estimated lower face arcs in the face’s coordinate system. Each arc has two patches inside and two patches outside the estimated boundary. It compares luminance high-pass prediction errors after tracking each patch to the two immediate native neighbors. The two same-side patch pairs control for local variation. Eligibility requires a single face, stable geometry, sufficient local texture and reliable forward-back tracking. A separate check excludes residuals concentrated near the native eight-pixel grid. The published method uses existing face detection weights and an independent implementation of the residual measurement. [Video-level blending](https://arxiv.org/abs/2408.17065) · [Spatial and temporal forensic residuals](https://arxiv.org/html/2503.20991v1) · [Explicit video features](https://arxiv.org/abs/2610.03380)

For each eligible arc, the signed log gap must exceed both flanks by at least log(1.5) in at least 60% of the centers within the selected interval, with a minimum of three. At least two arcs must be measurable to classify coverage. The comparison was frozen before scoring. A second research run would require local support in at least 18 swaps across four source groups, at least 80% measurement coverage across the 39 original flags and zero false support on the saved camera stress group. This point governs the local research decision. [Frozen specification](evaluation-panels/video-selected-face-boundary-freeze-2026-11-11.json)

## Results

| Cohort | Views | Selected intervals | Measured | Corroborated |
| --- | ---: | ---: | ---: | ---: |
| Known swaps from six RTFS sources | 36 | 36 | 4 | 0 |
| Matched RTFS originals and ordinary edits | 60 | 0 | — | — |
| Saved genuine camera stress source | 3 | 3 | 0 | — |
| Prepared genuine camera coverage source | 3 | 0 eligible | — | — |

All 36 known swap flags kept their original endpoints and had at least 0.50 intersection-over-union with the known edit interval. The four measurable swap windows came from one source and supplied no local corroboration. The three genuine stress views were unmeasurable, so this comparison provides no estimate of false support on genuine selected intervals. The 60 matched controls had no baseline flag to corroborate. [Exact aggregate](evaluation-panels/video-selected-face-boundary-2026-11-11.json)

All 410 selected center triplets passed the face geometry checks. Only 289 of 2,050 arc observations met the full patch requirements. The remaining 1,761 could not establish four locally reliable patches in both directions. None of the arcs that met the patch requirements was excluded by the codec-grid check. The bottleneck was local measurement availability, and the four measurable intervals produced no evidence of a repeatable edit seam. This version stays in research. There were no supporting intervals from which to generate a crop exhibit.

The local CPU run verified 558 native RGB frames and presentation timestamps in 274 seconds, with about 235 MiB peak process memory and 220 KiB of scratch output. A separate standard-library audit reproduced all 102 outcomes, 39 selected clocks, regional support counts and edit-overlap counts. Eight focused tests cover exact selection replay, uniform motion, unreliable geometry, codec-grid rejection, comparison patches and persistence.

## Reproduce

Use the retained video archive and the private baseline journal from the landmark reliability comparison. The runner reads the exact previous interval and writes its private measurements to a fresh directory. When the cue finds support, the private contact sheet shows the patch positions before, during and after the selected interval with their verified timestamps.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_selected_face_boundary freeze "$run_dir" --baseline "$baseline_run"
backend/.venv/bin/python -B -m scripts.video_selected_face_boundary run "$run_dir" --baseline "$baseline_run"
backend/.venv/bin/python -B -m scripts.video_selected_face_boundary readout "$run_dir" --baseline "$baseline_run"
```
