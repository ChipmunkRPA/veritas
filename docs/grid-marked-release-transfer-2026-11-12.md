# Earlier-copy timing with a shorter marked waveform

The production marked comparison localized **13 of 26** exact shifts. The shorter, picture-anchored comparison localized **25 of 26** under the same evidence rule. Both returned zero alerts on 32 unchanged or gain-only views. This is a transfer test on previously observed native clips and supports a new held test with independent speakers and edited clips. [Public counts and verification](evaluation-panels/grid-marked-release-transfer-2026-11-12.json)

## What changed

The two arms use the same production earlier-copy entry point, verified native clocks, shared-picture admission, mouth geometry and evidence assets. The supplied mark is the earlier last-closed native picture from a muted-video observation. Six lexical-negative controls use a fixed mark at picture 37. Neither arm receives the separately selected acoustic event. The production route searches for 160 ms of audio starting 40 ms after the mark, with a ±480 ms search and correlation and margin floors of 0.90 and 0.07. The candidate starts a 60 ms excerpt at the mark and uses the earlier reference experiment's ±400 ms search and floors of 0.94 and 0.08. The experiment compares those two fixed sets of parameters. [Earlier source-bound landmark and shift evaluation](grid-fresh-release-qualification-2026-11-12.md)

A correct location requires the signed estimate to be within two 25 fps pictures of the known offset and the entire reported audio excerpt to lie inside the edited range. Positive values mean the visible event moved later relative to sound. The 26 controlled shifts move only the soundtrack by 240 ms. They preserve every picture and all audio samples outside the authored range.

| Delivery | Views | Current assessed | Current localized | Candidate assessed | Candidate localized |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unchanged original | 19 | 13 | — | 13 | — |
| Local gain only | 13 | 13 | — | 13 | — |
| Audio moved earlier | 13 | 13 | 13 | 13 | 13 |
| Audio moved later | 13 | 0 | 0 | 13 | 12 |

All 13 earlier moves produced +240 ms in both arms. The candidate measured −240 ms for all 13 later moves. One reported 60 ms excerpt extended 120 samples, or 7.5 ms, beyond the edited interval, so that delivery does not count as a correct location. The earlier waveform-oracle experiment localized all 26 using a separately observed acoustic center. It is a different measurement.

The independent reread verified 182 picture, crop and PCM records directly from the source media and replayed 116 timing dispositions. Each assessed result reproduced the exact signed offset from the source picture and audio positions. No change was made to production behavior. The evidence measures relative timing in clock-matched copies and supplies zero weight to AI authorship confidence.

## Reproduce

Use the verified fresh-release study as `panel` and a new local directory as `run`. The two arms run on the retained landmark runtime and require the same model asset as the production local comparison.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze score; do
  backend/.venv/bin/python -B -m scripts.grid_marked_release_transfer "$phase" "$panel" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.grid_marked_release_transfer review "$panel" "$run" \
  --public "$run/public-aggregate.json"
```
