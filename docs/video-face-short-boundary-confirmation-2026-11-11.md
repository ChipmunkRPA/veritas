# Prospective confirmation of shorter face-appearance ranges

Applying dense timing only to flagged coarse ranges of one second or less improved overlap with the known edit in 19 of 36 swap views. No range worsened. The number with at least 0.75 temporal overlap rose from 21 to 36. All file flags stayed identical. [Exact aggregate and replay commitments](evaluation-panels/video-face-short-boundary-confirmation-2026-11-11.json)

## Fixed comparison

The previous study found that dense timing helped short coarse ranges and often worsened longer ones. This test froze the one-second rule before scoring. It used six new head-rotation source groups from the same archived publisher and two different genuine camera groups. Source links and known donor identifiers were distinct from both earlier boundary and texture trials. The new groups had participated in earlier whole-clip research. Three also carried the publisher's gesticulation tag. Eighteen exact original before, during and after pictures showed measurable variation in facial landmark geometry on all six recordings. The median variation in the nose-position proxy was 0.058 inter-eye distances. The proxy describes change in the observed picture geometry. [Earlier timing study](video-face-native-boundary-2026-11-11.md) · [Source and license review](video-common-source-preparation-v89.md)

The moving-source panel used two publisher-provenance swap families and matched originals, lighting, blur and time-step controls. Every intervention used exactly verified pictures and native presentation times. The short durations were nominally 0.75 and 1.50 seconds, with native half-open boundaries. The longer duration used the full approximately two-second native insert. The two genuine camera groups supplied unchanged, ordinary chroma-converted and recompressed views. The shorter delivery edits used the same native clock within the existing three-second camera edit. [Native source construction](video-native-inserts-v95-method.md) · [Genuine camera deliveries](video-vcd2-delivery-v206-method.md) · [Camera data license](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA)

The current single-face comparator uses its unchanged four-samples-per-second window and decision point. The prospective rule attempts the prior 12 Hz refinement only when the flagged coarse duration is at most exactly one second. Both boundaries must have two measured neighboring frames on each side, with consistent face-box overlap and verified clocks. The range can fall back to the original timing. This comparison runs the appearance calculation and pinned models on a single observed face. Full app tracking and signed report integration are separate.

| Known swap interval | Current ≥0.75 overlap | Fixed rule ≥0.75 overlap | Ranges improved |
| --- | ---: | ---: | ---: |
| 0.75 seconds | 4/12 | 12/12 | 12/12 |
| 1.50 seconds | 5/12 | 12/12 | 7/12 |
| Approximately 2 seconds | 12/12 | 12/12 | 0/12 |
| **Total** | **21/36** | **36/36** | **19/36** |

All 36 swaps were flagged at the same point in both arms, with at least 0.50 overlap. The 12 short gains covered all six moving source groups. The seven 1.50-second gains covered four groups whose flagged ranges measured one second. All 12 longer coarse ranges exceeded one second and kept their original timing. Median absolute onset and end errors fell from 0.133 and 0.147 seconds to 0.040 and 0 seconds for the 0.75-second views. For the 1.50-second views they fell from 0.200 and 0.280 seconds to 0.040 and 0 seconds. The repeated views share underlying source footage.

Both arms flagged 0 of 60 moving-source controls and 14 of 14 genuine camera control views. The latter came from two camera groups, including their unchanged inputs. Their existing flagged ranges measured 1.500 to 2.233 seconds and none entered refinement. The trial identifies a timing improvement. Existing camera false alarms still limit the appearance cue.

In one example, the known range was **2.280–3.760 seconds**, the original range was **2.480–3.480 seconds**, and the refined range was **2.320–3.760 seconds**. Independent command replay checked the exact clock and verified nine PNGs. The before, during and after pictures for that example were at **2.240, 3.000 and 3.760 seconds**. All 110 views were assessed. The CPU run verified 2,553 native picture inputs in 321.2 seconds, with 332 MB maximum worker RSS. Media and individual measurements stay private.

## Local app parity

An isolated prepaid Full-profile run used the actual app probe, shot selection, sampler, face tracker, native face detection and appearance adapter on one known 0.75-second swap and its matched original. It scheduled those three detector families only. The test used pinned CPU weights and a simulated payment provider. Both 150-picture inputs reproduced every native RGB frame, PTS, DTS and display duration. Each app run had one shot, one track, 24 sampled pictures and 12 appearance windows. The original had no flagged range.

The swap's app-selected range was **2.840–3.360 seconds**, a measured duration of 0.520 seconds. The known range was **2.640–3.400 seconds**. All 21 additional native samples bound to the same app-selected track. Offline refinement estimated **2.680–3.400 seconds**, improving overlap from 0.684 to 0.947. Its before, during and after PNGs matched source RGB at **2.600, 3.000 and 3.400 seconds**. The appearance score stayed identical.

The existing coarse packet matched the JSON export and verified in the signed package against the uploaded source. The package bytes and API evidence were unchanged after offline refinement. Its 17-page PDF and text exports passed the model-name scan. The initial harness expected an optional confidence-effect field missing from the coarse packet schema. The review signal had zero weight. Verification resumed from that completed analysis after correcting the assertion. [Aggregate app parity receipt](evaluation-panels/video-face-short-boundary-app-parity-2026-11-11.json)

## Opt-in app integration

The app now runs the same refinement on qualifying new Full and Deep analyses when `VERITAS_VIDEO_FACE_BOUNDARY_REFINEMENT=true` and the appearance review is selected. The ordinary evidence view, PDF and JSON keep the measured review range and show the refined estimate with its own exact before, during and after frame images and face-context crops. The timing estimate carries zero weight in the overall AI confidence. If both transitions cannot be measured within the bounded pass, the original range is retained. The deployment setting defaults to off.

The isolated prepaid Full replay again used the actual local tracking, face and appearance checks. Its known swap produced the same **2.840–3.360 s** measured range and confidence decision as the disabled run, and the app itself estimated **2.680–3.400 s** from 21 native inputs. All six corresponding frame and crop images matched both the native source pixels and PDF images. The API, JSON export and signed ZIP agreed, and offline verification decoded the original again. The matched original produced no flag or refinement. The harness simulated Stripe transport, and analysis remained unqueued until a valid paid event. [App integration receipt](evaluation-panels/video-face-short-boundary-app-integration-2026-11-11.json)

## Local replay

Run from the repository root with the exact archived media and both prior private run directories.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_face_short_boundary_confirmation freeze "$run_dir" --archive "$archive" --texture-run "$texture_run_dir" --boundary-run "$boundary_run_dir"
backend/.venv/bin/python -B -m scripts.video_face_short_boundary_confirmation run "$run_dir" --archive "$archive"
backend/.venv/bin/python -B -m scripts.video_face_short_boundary_confirmation readout "$run_dir"
```
