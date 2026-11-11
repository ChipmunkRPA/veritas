# Local color transitions at video face boundaries

The previous cheek and jaw check could measure only four of 39 selected intervals. Its local motion test needed all four patches to track to both adjacent frames. This comparison measures the color transition across the boundary within individual frames and uses a quorum of available cross-sections. It searches for a change in one face-relative arc across a short time range. [Previous motion test](video-selected-face-boundary-2026-11-11.md)

## Method

The runner estimates five lower face arcs from the detected face and eye axis. It measures seven cross-sections per arc at three offsets and two native pixel scales. A valid scale requires four in-bounds cross-sections. The cue measures the Lab chroma step across each cross-section, compares it with identically spaced steps farther inside and outside the arc, and requires agreement in the direction of the color change. It keeps the weaker of the two scales. The earlier whole-clip color check required bilateral color agreement and pooled a single response for the file. This trial evaluates unilateral, face-relative regions through time. [Earlier whole-clip color comparison](video-face-chroma-transfer-2026-11-11.md) · [Face-blending research](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Face_X-Ray_for_More_General_Face_Forgery_Detection_CVPR_2020_paper.html) · [Video-level blending research](https://arxiv.org/abs/2408.17065) · [October 2026 regional benchmark](https://arxiv.org/abs/2610.21438)

The scanner uses the nearest native pictures at four times per second and fixed windows of three, five, seven or eight samples. A range needs measurable left and right flanks and the same arc available on at least 60% of its core samples. Its score is the persistent excess over both flanks. The recorded end is the first sample after the half-open candidate window. The private score journal gives the arc and native coordinates for one frame before, during and after each selected window. The regional coordinates describe where the color change was measured. The native insert receipts provide edit-time truth and cannot verify AI-modified pixels.

The only model is the existing face detector. The cue has no training, motion tracking or reused research implementation. Before scoring, the runner freezes the method, source commitments and evaluation rules. It then seals the largest measured response from 31 older genuine controls before opening the held phase. A candidate flag must strictly exceed that point. The research advancement criteria require at least 12 of 24 correct localizations across at least four sources, no more than two false flags, and at least 80% assessed coverage in swaps and controls separately. A correct localization overlaps the known interval by at least 0.50 intersection-over-union. [Frozen method](evaluation-panels/video-chroma-boundary-transfer-freeze-2026-11-11.json) · [Face detector terms](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

## Sources

The held matched set has six RTFS source groups outside the four earlier localizer rosters, with distinct known source links and donor keys. They supplied 24 swaps across two publisher families and two exact edit lengths, along with 42 genuine and ordinary control views. The ordinary interventions include lighting, blur and a two-frame source-content shift inside the fixed video clock. Two other genuine camera sources supply six unchanged, chroma-converted and recompressed deliveries. The development set uses four older RTFS sources and one different camera source. The held sources are new to this measurement and have previous project-level whole-clip or app exposure. The original dataset uses archived publisher rights and has incomplete donor identity linkage. [RTFS rights and source scope](video-common-source-preparation-v89.md) · [Exact native inserts](video-native-inserts-v95-method.md) · [Camera data terms](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA)

## Results

| Held cohort | Views | Assessed | Flagged | Correct edit interval |
| --- | ---: | ---: | ---: | ---: |
| Known swaps | 24 | 24 | 0 | 0 |
| RTFS originals and ordinary edits | 42 | 42 | 0 | — |
| Genuine camera deliveries | 6 | 6 | 0 | — |

All 31 development controls were measurable. Their largest response was 1.9002143144607544. The runner sealed that point before opening the held phase. The highest held swap scored 1.031 and the highest held control scored 1.208. Independently of the sealed point, the top interval overlapped the known edit by at least 0.50 in only five of the 24 swaps. The assessed swap-versus-control score AUC was 0.572. The new cue had complete measured coverage on this panel and insufficient discrimination. No case yielded supporting crop evidence. The check stays experimental. [Aggregate and sealed point](evaluation-panels/video-chroma-boundary-transfer-2026-11-11.json)

The local CPU runs verified 945 native RGB pictures and timestamps in a combined 493 seconds, with about 223 MiB peak worker memory. A standard-library audit recomputed the sealed point, counts, interval overlaps and 216 held frame-anchor checks. Six focused tests cover the color transition, smooth ramp, partially visible arc, inconsistent color direction, missing temporal samples and unchanged intervals. No video or per-source measurement is included in the public receipt.

## Reproduce

Use the retained archive and the private run directories of the four previous localization studies. The source and measurement journals stay local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_chroma_boundary_transfer freeze "$run_dir" \
  --prior "$texture_run" --prior "$boundary_run" --prior "$short_boundary_run" --prior "$landmark_run"
for stage in development seal held readout; do
  backend/.venv/bin/python -B -m scripts.video_chroma_boundary_transfer "$stage" "$run_dir" || break
done
```
