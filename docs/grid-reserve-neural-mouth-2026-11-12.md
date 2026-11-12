# Fresh neural mouth redraws with verified source timing

## Fixed sources and rendering

Select the first usable original in each of two separate reserve groups, one with a direct publisher word-directory match and one with a verified remapping. The source choice uses the previously verified word clocks and original picture geometry. Each chosen original has a visible closure and reopening near a word whose initial sound normally uses both lips. The source groups were outside the earlier neural pilot. The original waveforms and word labels were qualified previously. [Reserve word and waveform audit](grid-reserve-word-clocks-2026-11-12.md) · [GRID source terms](https://zenodo.org/records/3625687)

Freeze a 16-picture native interval starting three pictures before the measured closure. Both neural conditions use the exact source picture at each output time for identity and geometry. One uses the expression observed on that source picture. The other uses the same speaker's expression exactly eight native pictures earlier, or 320 ms. A separate ordinary control uses earlier source pixels instead of neural output, with the same crop, alpha and timing. All deliveries preserve the complete original soundtrack. The two source-time constructions test whether a genuine mouth movement can remain recognizable after neural reconstruction. The earlier-expression and ordinary cases check a known timed intervention.

The renderer uses the retained, rights-reviewed LivePortrait human animation checkpoints on CPU float32. The existing bounded convolution implementation keeps the full 512-pixel output and model arithmetic. Each rendered image is inverse-mapped into the original frame with its native mouth alpha, half opacity at the two edit edges, and full opacity inside. The renderer and observer work in a new scoped reserve context. Historical pilot-only locks and source files remain intact. [LivePortrait paper](https://arxiv.org/abs/2407.03168) · [Pinned MIT licence](https://github.com/KlingAIResearch/LivePortrait/blob/9b294b3d0536135442ea73cb01e6cb3ca7029dd3/LICENSE) · [Previously verified renderer](grid-portrait-frames-v146-result.md)

## Independent timing and source evidence

The verifier re-decodes every submitted picture and the whole native PCM independently after construction. It verifies actual changed-picture masks, exact presentation timestamps and unchanged soundtrack samples. A separate stateless landmark pass runs on the finished decoded pictures, using the same rights-reviewed geometry task that registered the original crops. It compares a seven-picture mouth-opening trajectory against the genuine event, after subtracting the mean opening. It selects the best observed closure and reopening within the edit without using the intended offset. An event within one native picture of the genuine or 320 ms later closure supports the corresponding timing. Missing or different measurements remain explicit. [Native aperture measurement](grid-achieved-motion-validation-design.md) · [Face Landmarker reference](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)

The closure observer measures visible motion at 40 ms resolution. Publisher word intervals and exact audio projection support correspondence, while phone boundaries and perceptual synchrony across the full clip remain unverified. The audit retains original before/during/after evidence and aligned canonical PCM locally.

## Observed results

| Reserve source | Exact edited interval | Original closure | Source-time neural | Earlier-expression neural | Earlier-pixel ordinary |
| --- | --- | --- | --- | --- | --- |
| First group | 0.400–1.040 s | 0.520 s | 0.520 s | 0.840 s | 0.840 s |
| Second group | 0.680–1.320 s | 0.800 s | 0.800 s | 1.120 s | 1.120 s |

Both source-time redraws retained the original closure picture exactly. Both earlier-expression redraws and both ordinary controls moved that event eight pictures later while preserving the sound. For the first source-time redraw, the before, during and after pictures are **0.360**, **0.520** and **1.040 s**, with **10,240** aligned source audio samples over the edited interval.

The CPU run generated 64 neural pictures in 223.4 seconds, with five strict tensor-exact checkpoint loads, 320 generator root forwards and 2.51 GiB peak reported RSS. The independent review verified 450 delivered native pictures, 96 exact edited-picture masks and 285,888 unchanged soundtrack samples. All six edited intervals replayed against the original clips. The initial sandboxed geometry setup could not initialize a macOS graphics context. The isolated local run completed with the same CPU task and settings. A metadata-only correction to the private group labels was independently replayed and left every measured value and timestamp identical. [Aggregate and source replay](evaluation-panels/grid-reserve-neural-mouth-2026-11-12.json)

## Reproduce

Use the retained qualified reserve sources and the existing licensed renderer and geometry assets. Run on a host where the local macOS geometry task can initialize its graphics context. Media and per-case evidence stay local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.grid_reserve_neural_mouth preflight "$qualification" "$run"
backend/.venv/bin/python -B -m scripts.review_grid_reserve_neural_mouth freeze "$qualification" "$run"
backend/.venv/bin/python -B -m scripts.grid_reserve_neural_mouth render "$qualification" "$run"
backend/.venv/bin/python -B -m scripts.review_grid_reserve_neural_mouth replay "$qualification" "$run"
```
