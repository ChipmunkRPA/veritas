# Timing and visual checks on the new mouth redraws

The existing timing check and the previously frozen short audiovisual candidate returned zero ranges on eight native clips. Both missed the measured 320 ms delays. The two experimental visual-detail checks missed both synchronized neural redraws. They located a delayed neural edit and an ordinary retime in one source group. The broad-motion adjustment also flagged that group's untouched original. [Readout and native evidence](evaluation-panels/grid-reserve-neural-mouth-readout-2026-11-12.json)

## Comparison

Use the two reserve originals, their two source-time neural redraws, two earlier-expression neural redraws and two earlier-pixel ordinary retimes. Independent source replay verified the changed pixels and native picture clocks. The visible closure moves eight pictures later in each earlier-driven view while the original PCM stays unchanged. Both originals were held in the earlier audiovisual study and did not enter its fit or point selection. The four newly rendered clips had no prior detector exposure. [New source and closure evidence](grid-reserve-neural-mouth-2026-11-12.md)

The current timing comparison applies the same single-face source-bound measurement and serving timing arithmetic as the prior study. The short audiovisual arm uses the prior fitted interaction matrix and review point. Signed localization requires at least 0.40 time intersection over union and an audio-minus-picture lag estimate within two native ticks of the known shift. The visual arms use the earlier independent frozen fine-detail points and require at least 0.50 time intersection over union with the exact edited interval. No point, model, time window or inclusion rule was fitted to the new results. [Earlier timing comparison](video-lipsync-local-contrast-transfer-2026-11-12.md) · [Earlier visual comparison](video-mouth-temporal-detail-2026-11-12.md)

## Outcomes

| Group | Current timing ranges | Short audiovisual ranges | Fine-detail useful ranges | Broad-motion adjusted useful ranges |
| --- | ---: | ---: | ---: | ---: |
| Untouched originals | 0/2 | 0/2 | 0/2 flagged | 1/2 flagged |
| Source-time neural redraws | 0/2 | 0/2 | 0/2 | 0/2 |
| Earlier-expression neural redraws | 0/2 | 0/2 | 1/2 | 1/2 |
| Earlier-pixel ordinary retimes | 0/2 | 0/2 | 1/2 | 1/2 |

All eight clips had 74 verified mouth crops. The current comparison measured one timing window per clip, too few for a local timing-change finding. The audiovisual candidate assessed 36 or 43 brackets per clip and produced no flags at its frozen point.

The two visual arms localized **0.400–1.120 s** on the first earlier-expression redraw. Its authored interval was **0.400–1.040 s**. The same source's ordinary mouth retime produced ranges of **0.400–1.120 s** and **0.480–0.880 s** respectively. The adjusted visual arm also flagged **0.400–1.120 s** on the untouched original. An ordinary retime is a genuine timing change and a control for AI attribution.

The readout replay matched all eight file hashes and 15 exact native crop references. A repeat produced byte-identical detector results. This small panel supplies clear missed cases for later checks and carries no new accuracy or serving claim. [Frozen readout receipt](evaluation-panels/grid-reserve-neural-mouth-readout-2026-11-12.json)

## Reproduce

Use the completed local neural source bank and the retained earlier timing and visual calibration files.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.grid_reserve_neural_mouth_readout freeze "$study" "$readout"
backend/.venv/bin/python -B -m scripts.grid_reserve_neural_mouth_readout run "$study" "$readout"
```
