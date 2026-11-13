# Short speech windows with wider context

This comparison measures whether two-second windows can recover narrower
locations within noisy recordings already measured with three-second windows.
It independently implements duration-specific checks with the installed
voice classifier and no new weights. The underlying classifier was trained
to score complete speech inputs. Related work separates locating candidate
intervals from scoring their authenticity and reports difficulty transferring
scores across recording conditions. [Speech inpainting and boundary refinement](https://arxiv.org/html/2605.02223v1) ·
[Temporal pyramid results](https://arxiv.org/html/2606.16837v1) ·
[Installed checkpoint terms](commercial-model-audit.md)

## Fixed comparison

The input bytes, selection and held groups, and three-second responses come
from the [staggered-window transfer](audio-noise-window-transfer-2026-11-12.md).
All 134 waveforms were previously measured at three seconds. This study
adds new two-second measurements on the same audio. The 18 known edited
source groups cover six generators. Their speakers had earlier project
exposure. The twelve genuine selection speakers differ from the twelve
held genuine speakers, and all differ from the edited speakers. Restaurant
and coastal street recordings occur only in selection, and market and
traffic recordings occur only in held views. The held foreground is 5 dB
quieter than the added sound. The mixture and gain-only reference use the
same peak-safe gain. Isolated excerpts of each sound remain controls.

Score native two-second intervals every 0.75 seconds, plus an exact
end-aligned window when needed. A window needs speech across at least half
its duration using the existing local speech activity check on the original
recording. Each speaker contributes its maximum score to each duration's
ordinary selection distribution. Compare short responses alone and short
responses supported by a three-second window containing all the same
samples. In the supported arm, both duration-specific scores must exceed
their respective second-highest genuine selection maxima. The score ranks
use strict comparisons to the genuine maxima. The wider response can use
any of the already recorded three-second positions, including the
staggered grid. The short score uses log-odds. The stored wider responses
use raw probabilities despite the freeze's logit label. The method only
ranks those scores, so converting them to log-odds gives the same order.
Require the same combined rank to exceed the strongest
background-only rank. Each arm reports up to three intervals in descending
rank order, suppressing any that share at least half their length with an
earlier selection. Ties prefer the earlier start. The fitted points were
recorded after the 40 selection deliveries, before the 94 held deliveries
were scored at two seconds.

The comparison includes the original sparse and dense results. The
combined arm preserves every dense range and adds supported short
intervals. A useful range must cover at least half the known edit with
temporal intersection over union of at least 0.30. Count genuine time
inside and outside edited files, deduplicating overlapping ranges. Follow-up
requires three added noisy edit groups across two families, up to one
additional noisy genuine group, up to two additional percentage points of
genuine time under noise, and no added clean or isolated flags. The
comparison uses scalar scores and does not fit new overall confidence.

## Held results

Each cell shows useful locations on edited files or flags on genuine files.

| Delivery | Sparse | Dense | Short | Short with context | Dense plus short with context |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gain-only edited, 18 | 4 | 11 | 10 | 6 | 12 |
| Market edited, 18 | 2 | 3 | 0 | 0 | 3 |
| Traffic edited, 18 | 2 | 3 | 0 | 0 | 3 |
| Gain-only genuine, 12 | 0 | 0 | 0 | 0 | 0 |
| Market genuine, 12 | 2 | 3 | 0 | 0 | 3 |
| Traffic genuine, 12 | 0 | 0 | 0 | 0 | 0 |
| Isolated backgrounds, 4 | 0 | 0 | 0 | 0 | 0 |

The short check measured 999 of 1,074 planned held windows. Every one of
the 90 speech deliveries had a response. The four isolated backgrounds
were unassessed. Under loud noise the additional short windows supplied
no useful locations, with or without wider context. The combined method
preserved all six noisy dense locations, the same three genuine market
flags, and the same 17.61 of 502.36 seconds of genuine audio highlighted.

On gain-only speech the combined arm recovered one additional known edit
without an added genuine file flag. The noisy result missed both gain
requirements. The combination stays outside the app. The gain-only finding
supports testing on genuinely new voices before adding a shorter check to
longer clean recordings. [Held receipt](evaluation-panels/audio-short-coarse-transfer-2026-11-12.json)

## Replay

The portable reader checks every duration and sample range, reconstructs
the selection cutoff and held results, and compares the unchanged
three-second receipt. Supplying the retained native run also checks every
short-window PCM commitment. The review uses the stored model responses.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.audio_short_coarse_transfer review \
  docs/evaluation-panels/audio-short-coarse-transfer-2026-11-12.json
PYTHONPATH=backend python3 -B -S -m scripts.audio_short_coarse_transfer review \
  "$study/receipt.json" --source "$earlier_run"
```

To repeat inference, first recreate the earlier native run with the exact
publisher ranges, sound recordings and checkpoints in the
[source instructions](audio-noise-window-transfer-2026-11-12.md#replay).
Set the environment below and use a fresh destination.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export VERITAS_AUDIO_DETECTOR_MODE=candidate VERITAS_BILLING_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
backend/.venv/bin/python -B -m scripts.audio_short_coarse_transfer measure \
  "$study" --source "$earlier_run"
```
