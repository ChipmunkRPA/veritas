# Persistent detail around short mouth redraws

The candidate localized four of 20 CREMA-D redraws, compared with one for the earlier direct visual check. On GRID it localized one of 32 redraws, the same count as the earlier check. It missed all nine redraws across GRID and the separate reserve with independently verified source-time-compatible mouth movement. One of four ordinary hard cuts triggered the candidate. It remains experimental. [Frozen results and source anchors](evaluation-panels/video-mouth-sector-persistence-2026-11-12.json)

The two independently verified reserve redraws preserved the original closure picture at 0.520 and 0.800 seconds, respectively, and kept the original audio. The observer measures a visible event every 40 ms. These clips do not independently establish phoneme-level or listener-perceived synchrony. Both originals have 75 native pictures at exact 25 fps and are MPEG-1. The rendered and conventional controls have the same display and cadence in lossless FFV1. Each has 47,648 verified 16 kHz samples on one continuous audio clock. The released static-mouth check has no coverage on these codec, cadence and duration profiles. The previous default timing calculation measured one window and returned no timing range on each of the eight clips. [Reserve source qualification](grid-reserve-neural-mouth-2026-11-12.md) · [Earlier timing and visual readout](grid-reserve-neural-mouth-readout-2026-11-12.md)

## Native comparison

The candidate measures signed curvature in chroma and fine-detail energy across four adjacent bands. Three ellipses centered on the observed mouth define eight angular sectors at each size. The calculation stays on the decoded native pixel grid and subtracts measurements from a reference patch above the mouth. The primary arm requires a consistent change in the same three or more sectors and the same direction and channel in at least three quarters of an interval's pictures. Both adjacent flanks must support the comparison. The separate arm permits the contributing sectors to change during the interval. This independently tests a specific aspect of regional visual evidence motivated by recent research. A competing November preprint on compositing had been withdrawn over overlapping train and test sources and was excluded from the evidence for this study. [SAVe regional self-blending research](https://arxiv.org/abs/2603.25140) · [MVFNet spatial and temporal residual research](https://arxiv.org/abs/2503.20991)

The third-largest ordinary maximum set each point on 56 development views from four known GRID components. Held GRID uses four different components. CREMA-D confirmation uses ten other groups. An additional eight ordinary recompressions and four hard cuts test delivery effects. The separate reserve comprises two further source groups. The retained media had earlier project exposure. Controls include blur, resampling, brightness edits, retiming, original pictures and audio-only changes. No new weights, media or renderer were added. A useful location must overlap at least half of the authored interval by temporal intersection over union. [Held source roles and controls](video-mouth-jaw-transfer-2026-11-11.md) · [GRID publisher terms](https://zenodo.org/records/3625687) · [CREMA-D publisher terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt)

| Held outcome | Earlier direct visual check | Independent sectors | Persistent sectors |
| --- | ---: | ---: | ---: |
| GRID correct redraw ranges | 1/32 | 0/32 | 1/32 |
| GRID ordinary flags | 0/56 | 0/56 | 0/56 |
| CREMA-D correct redraw ranges | 1/20 | 0/20 | 4/20 |
| CREMA-D ordinary flags | 8/40 | 4/40 | 2/40 |
| H.264 re-encode flags | 0/8 | 0/8 | 0/8 |
| Genuine hard-cut flags | 0/4 | 0/4 | 1/4 |
| Separate reserve correct redraw ranges | — | 0/4 | 0/4 |
| Separate reserve ordinary flags | — | 0/4 | 0/4 |

The CREMA-D additions span two source groups and replace the direct check's sole correct location. The two ordinary alerts both occur on blur controls from one of those groups. One useful range was **0.827–1.395 s** for a redraw authored at **0.694–1.362 s**. It uses the original frames at indices 22, 31 and 41 before, during and after the measured range. The candidate measured 80/88 GRID views, all 60 CREMA-D views, all 12 delivery controls and 4/8 reserve views. The primary arm met its CREMA-D and coverage gates, but missed the required GRID and compatible redraw counts and added an ordinary hard-cut flag. The separately decoded native audit verified 13 files, 923 native presentation timestamps and 39 exact original frames and mouth crops. The audit verifies displayed source observations. The candidate does not establish which pixels a renderer changed.

## Replay

Use a new local `$run` directory and the retained source archive. The runner binds the source banks and freezes the numerical rules before measuring held or reserve cases. The public comparison arithmetic requires only Python's standard library.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze development calibrate held reserve; do
  backend/.venv/bin/python -B -m scripts.video_mouth_sector_persistence "$phase" "$run" || exit 1
done
python3 -B -S -m scripts.review_video_mouth_sector_persistence docs/evaluation-panels/video-mouth-sector-persistence-2026-11-12.json
```
