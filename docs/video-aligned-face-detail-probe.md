# Local face detail and motion comparison

The comparison checks whether two inexpensive measurements can identify where a known face swap occurred. One measures a jump in image detail near the face outline. The other measures how much the inner face changes across neighboring frames after aligning the face and subtracting a shared color change. The October 2026 illumination study tests a related distinction between a useful visual cue and a transferable detector operating point. This comparison uses its own measurements and no assets from the study. [Illumination study](https://arxiv.org/abs/2610.11706)

## Method

The runner selects 12 source groups from each of the existing development and confirmation rosters by a fixed salted source-group order. The rosters have disjoint source IDs. Every group supplies one original, one ordinary edit, and two swaps from different generators. These clips were previously used to evaluate the app's appearance checks. This evaluation introduces new measurements, and it reuses their saved face landmarks and exact source bindings. The dataset publisher declares CC BY-SA 4.0. The public aggregate excludes media, faces, and per-case identifiers. [Publisher dataset card](https://huggingface.co/datasets/kontur-ai/rtfs-10k) · [Development method](video-v16-sface-development-protocol.md) · [Confirmation method](video-v17-sface-confirmation-protocol.md)

For each clip, the runner takes up to 16 evenly spaced centers with exact neighboring source frames. It verifies each source file, saved input record, native timestamp, decoded pixels and aligned crop. On the aligned crop, the detail measurement compares the median residual from a Gaussian blur in a band near the face outline with the two adjacent bands. The motion measurement subtracts the median color shift for each comparison and takes the smaller inner-to-outer residual ratio from the two neighbors. The reference uses the app's existing face-to-background noise measurement, expressed as absolute log distance from equal noise. All three arms use the largest measured value and its source time.

Each arm sets its operating point at the 90th percentile of the 24 development control maxima, with equality excluded. A location counts when the maximum crosses that point inside the known half-open edit interval. The comparison checks a time point. Whole-file confidence and review intervals require separate validation.

## Result

All **96 clips** and **3,168 native frame observations** were verified. The runner made zero new neural calls. A second execution reproduced every per-case measurement and count. The 48 confirmation clips had the following outcomes. [Exact aggregate](evaluation-panels/video-aligned-face-detail-probe.json)

| Measurement | Correctly timed swap flags / 24 | Missed swap locations / 24 | Original and ordinary-edit flags / 24 |
| --- | ---: | ---: | ---: |
| Existing face/background noise measurement | 1 | 23 | 0 |
| Detail near the aligned face outline | 2 | 22 | 5 |
| Motion inside the aligned face | 3 | 21 | 9 |

The outline and motion measurements added one and two correctly timed flags respectively and produced more false alarms. Both remain experimental. The production appearance check is a separate method and was not rerun in this comparison.

Two additional paired face-swap sources were considered. Science Data Bank restricts the Fakes for All package to noncommercial research and approval. DirectSwap describes a research-only agreement for its paired data and checkpoints. Neither package was used. [Fakes for All access conditions](https://doi.org/10.57760/sciencedb.28605) · [DirectSwap release terms](https://arxiv.org/html/2512.09417v2#A6)

## Local reproduction

The runner needs the prepared v16 and v17 source and saved-frame receipts under `backend/data/independent-evaluation`. Run from the repository root with the established local decoder and Python environment. Each invocation uses a new trial directory.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
trial="$(mktemp -d)/trial"
backend/.venv/bin/python -B -m scripts.video_face_rim_probe freeze "$trial"
backend/.venv/bin/python -B -m scripts.video_face_rim_probe run "$trial"
```

Share `aggregate.json` for comparison. The trial's input plan and per-case observations contain private bindings.
