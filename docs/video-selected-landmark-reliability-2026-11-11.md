# Landmark reliability at the selected video interval

The original six webcam false flags that motivated this comparison had a sharp drop in inter-eye distance relative to face-box width. Selecting a lower-ranked interval simply moved those flags. This trial applies the existing 60% support rule to the incumbent's original selected interval. Weak support withholds that interval. Unavailable geometry is recorded separately and the original interval is retained. The test compares evidence reliability and supplies no verdict about the source file. [Earlier webcam measurements](video-saved-landmark-support-v254-result.md) · [Frozen rule](video-landmark-rerank-v259-method.md) · [New freeze](evaluation-panels/video-selected-landmark-reliability-freeze-2026-11-11.json)

## Fixed comparison

The panel contains six RTFS source groups with distinct known original links and donor keys. The six groups are outside the previous texture and short-timing studies and have prior whole-clip project exposure. Each contributes the unchanged view, two swap families and three ordinary edit types. Each edit is replayed at 0.75 seconds, 1.50 seconds and the complete verified native insert. The runner reproduces the prior 4 Hz single-face appearance comparison and selects the original top window at the frozen point. All frames used for selection and landmark support must match the original presentation clock and prior native RGB hashes. [Source and rights review](video-common-source-preparation-v89.md) · [Prior 4 Hz comparison](video-face-short-boundary-confirmation-2026-11-11.md)

The camera cohort has two independent source recordings with three delivery variants apiece. One triplet replays the saved full-app geometry and clock from v202, using its original selected records. The second runs the single-face appearance comparison on the three previously unscored v290 deliveries. The frozen rule requires two of the three known camera false flags withheld, zero losses among swap locations with intersection-over-union at least 0.50, and measured landmark geometry on at least 90% of selected windows. Those criteria would advance only the local pilot. [Saved camera panel](video-vcd2-full-app-v202-method.md) · [Prepared camera controls](video-vcd2-matched-controls-v290-result.md) · [Camera data terms](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA)

The geometry measurement uses the original consecutive frames around each center. It compares the median eye spacing divided by face-box width for the middle group with the average of the same medians from the two flanks. The two pinned vision models have MIT and Apache-2.0 terms, respectively. The geometry replay on the saved camera triplet invokes no model. [Face-detection license](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE) · [Face-representation license scope](https://raw.githubusercontent.com/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_recognition_sface/README.md)

## Results

| Cohort | Source groups | Planned views | Original flags | Flags withheld |
| --- | ---: | ---: | ---: | ---: |
| Known swaps | 6 | 36 | 36 | 0 |
| RTFS originals and ordinary edits | 6 matched to the swaps | 60 | 0 | 0 |
| Saved genuine camera triplet | 1 | 3 | 3 | 0 |
| Prepared genuine camera triplet | 1 | 3 | Unassessed | Unassessed |

All 36 swap ranges reached at least 0.50 intersection-over-union with the exact edit interval and kept their original endpoints. All 39 selected flags had measurable landmark geometry. The saved camera false flags all pointed to 7.0–8.3 seconds, outside the ordinary edit interval of 3–6 seconds. Their during-to-flank eye-width ratio was 1.004 in all three deliveries. The three prepared camera views had no eligible single-face appearance window, so they provide no false-flag estimate. [Exact aggregate](evaluation-panels/video-selected-landmark-reliability-2026-11-11.json)

The CPU run verified 1,494 native frames in 440.6 seconds with 333 MB peak process memory. A separate standard-library calculation reproduced all 102 decisions, interval intersections, coverage and group counts from the journal. The trial failed the false-flag condition. The earlier false windows had severe landmark collapse. The held genuine camera source kept stable eye spacing, so this rule cannot account for its false flags. This rule stays experimental.

## Reproduce

Use the same retained archive and the three earlier study directories. The commands write local evaluation outputs into a fresh directory.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_selected_landmark_reliability freeze "$run_dir" \
  --texture "$texture_run" --boundary "$boundary_run" --short-boundary "$short_boundary_run"
backend/.venv/bin/python -B -m scripts.video_selected_landmark_reliability run "$run_dir"
backend/.venv/bin/python -B -m scripts.video_selected_landmark_reliability readout "$run_dir"
```
