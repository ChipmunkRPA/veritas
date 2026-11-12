# Signed mouth and speech timing with an earlier copy

A paired timing check found 11 of 12 known offsets on four separately held source groups. It returned no ranges on 12 unchanged or coherently edited controls. The check requires an earlier native copy and a marked visible closure in that copy. The result describes movement relative to sound and carries no AI confidence. [Held readout and native replay](evaluation-panels/grid-paired-closure-timing-2026-11-12.json)

## Fixed method

Use the earlier copy's marked closed mouth and its measured reopening as the reference. Compare the seven-picture eye-normalized inner-lip aperture shape with nearby native pictures in the submitted clip. Require a good match and separation from substantially different alternatives. Independently match 160 ms of the earlier copy's actual waveform, beginning 40 ms after the marked event, to the submitted waveform. Require a distinct, strong waveform match. The signed result is the observed picture shift minus the audio shift. A positive value means the picture event moved later relative to its speech. Preserve both selected native picture timestamps, both exact PCM sample intervals and the range between the two event times.

The check uses time evidence at a visible articulation event. Speech releases and voicing can occur at different times. This test measures relative correspondence between two copies. It supplies no phoneme timestamp or synchrony guarantee for the whole clip. [Stop timing research](https://arxiv.org/abs/2606.28857) · [Verified source word clocks](grid-reserve-word-clocks-2026-11-12.md)

Discovery used the two completed reserve redraw pairs. Both known neural and both ordinary 320 ms visual delays were recovered exactly. The two source-time neural redraws and the two genuine recordings returned zero relative shift. These discovery recordings are separate from the four held groups. [Discovery sources and verified closure times](grid-reserve-neural-mouth-2026-11-12.md)

Freeze the existing method and select four other reserve groups using original native word clocks and visible closures only. Each supplies the unchanged recording and exact matched conventional controls. The edited controls move only the mouth 320 ms earlier or later, move only sound 320 ms later, move the mouth and sound together, or change local mouth brightness and sound gain. Every authored intervention is independently re-decoded. The target timing sign uses the measured closure shift and exact PCM change. The reference geometry uses the retained rights-reviewed stateless face task. A useful event estimate needs the correct sign and a lag within two pictures. The frozen gate requires eight useful estimates across at least three groups, at least 75% assessed coverage and zero original or matched timing-control flags.

## Held results

| Condition | Assessed | Paired timing ranges | Correct signed estimates |
| --- | ---: | ---: | ---: |
| Mouth 320 ms later | 4/4 | 4 | 4 |
| Mouth 320 ms earlier | 3/4 | 3 | 3 |
| Audio 320 ms later | 4/4 | 4 | 4 |
| Untouched original | 4/4 | 0 | — |
| Coherent mouth and audio retime | 4/4 | 0 | — |
| Local brightness and gain | 4/4 | 0 | — |

All 11 reported signed estimates were exact. In the missed early-mouth case, the measured native closure moved eight pictures, while the shape comparison exceeded its frozen support limit. The current single-face timing arithmetic measured all 24 cases and returned zero ranges. The earlier experimental single-file audiovisual candidate also measured all cases. It returned four ranges, including an untouched original and a brightness/gain control, with zero correct signed localizations.

Independent replay verified **1,800 native pictures**, **1,143,552 source-bound PCM samples**, **23 reported picture anchors** and **46 matched speech slices**. One correctly located ordinary visual retime moved the reference closure from **0.720 s** to **1.040 s** relative to the same speech, for **+320 ms**. Each matched speech slice contained **2,560 samples**. A timing difference is compatible with ordinary editing and says nothing by itself about how the pictures were generated.

The held cases use a short, exact 25 fps GRID profile and verified matching earlier sources. The neural examples are from discovery. Separate commercial codecs, independently encoded earlier copies and single-file detection remain outside this measured scope.

## Reproduce

Use the retained reserve timing qualification, completed discovery redraw study and existing timing calibration. The local MediaPipe runtime must be able to initialize its macOS graphics context. Media, native image evidence and per-source samples stay local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in discovery freeze author score; do
  backend/.venv/bin/python -B -m scripts.grid_paired_closure_panel "$phase" "$qualification" "$study" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_grid_paired_closure_panel "$run"
```
