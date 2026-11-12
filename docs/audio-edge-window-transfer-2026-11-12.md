# Short voice windows under competing speech

None of the four grids flagged a generated edit at its frozen ordinary-calibration point. The two-second Edge and Combined grids each ranked two useful edit locations above every genuine group in the held set. That retrospective gain supports a separate calibration replication.

## Held results

| Result | Full | Edge | Shifted | Combined |
| --- | ---: | ---: | ---: | ---: |
| Assessed files of 26 | 26 | 26 | 26 | 26 |
| Edits with any useful admitted window of 6 | 0 | 4 | 4 | 6 |
| Edits whose strongest window was useful of 6 | 0 | 2 | 3 | 3 |
| Generated flags at frozen point of 6 | 0 | 0 | 0 | 0 |
| Genuine false flags at frozen point of 20 | 0 | 0 | 1 | 0 |
| Genuine false groups at frozen point of 8 | 0 | 0 | 1 | 0 |
| Useful locations at 0 / 1 / 2 retrospective false groups | 0 / 0 / 0 | 2 / 2 / 2 | 0 / 2 / 2 | 2 / 2 / 3 |

At zero retrospective false groups, the Edge and Combined grids selected 0–2 seconds for a 0.16–0.80 second clone edit and for a 1.28–2.24 second stock edit. At two false groups, Combined also selected 3–5 seconds for a 3.44–4.24 second clone edit. Neither grid correctly localized the late edge edit. All six edits had at least one useful Combined window, so the remaining errors include which window scored strongest and whether its score cleared the ordinary calibration point. The matched false-group budgets use the held controls and describe ranking on this panel. The short-window check remains experimental. [Anonymous result ledger](evaluation-panels/audio-edge-window-transfer-2026-11-12.json)

The earlier call-processing study published a two-second research point in the same raw log-odds units. Applying it unchanged to these four grids gives a separate cross-grid comparison.

| Result at the earlier research point | Full | Edge | Shifted | Combined |
| --- | ---: | ---: | ---: | ---: |
| Generated flags of 6 | 1 | 3 | 2 | 3 |
| Useful locations of 6 | 0 | 2 | 2 | 3 |
| Genuine false files of 20 | 3 | 6 | 4 | 8 |
| Genuine false groups of 8 | 1 | 3 | 2 | 3 |

The Combined grid found three locations at the earlier point and flagged two more genuine groups than Full. It tested five candidate intervals per clip, which changes the distribution of its strongest response. A prospective replication can preselect a genuine-group false-alarm budget, fit the Combined grid on separate ordinary controls at that budget, and assess locations on another held cohort. [Earlier frozen point](audio-communication-transfer-2026-11-12.md)

## Fixed window comparison

The Full voice adapter plans 0–4 and 2–6 seconds on continuous six-second audio. A complete 0.64–0.96 second insert has at most 0.16–0.24 intersection over union with a four-second range. The earlier two-second grid starts at 1 and 3 seconds, leaving the two clip edges outside its coverage. An earlier overlap study tested the same window lengths on six held noisy inserts, all without a flag at its frozen points. This study uses newly assembled clips with known edge and between-window edits. [Full adapter comparison](audio-communication-serving-grid-2026-11-12.md) · [Earlier overlap study](audio-overlap-window-transfer-2026-11-12.md)

| Grid | Source windows in seconds |
| --- | --- |
| Full | 0–4, 2–6 |
| Edge | 0–2, 2–4, 4–6 |
| Shifted | 1–3, 3–5 |
| Combined | All five two-second windows |

The commercially admitted voice checkpoint processes each window with the usual input preprocessing. All grids share one speech-activity pass per exact recording and admit a window when at least half contains detected speech. Each grid ranks an input by its strongest admitted raw response and uses the earliest window if tied. The Combined grid adds window positions with the same checkpoint and underlying audio. [Model terms](commercial-model-audit.md)

For each grid, fit one comparison point to the maximum file score among 40 separately sealed genuine clips. They include untouched hosts, ordinary same-reader and other-reader splices, and isolated room recordings across 16 groups. All four points were written before held inference. An experimental flag needs a score strictly above its point. A false group counts once when any genuine control in the group flags. A useful location needs at least 0.30 temporal intersection over union and must cover at least half the edit. Retrospective matched false-group budgets show how the grids ranked the same held controls. [Calibration cohort provenance](audio-remote-neighbor-replication-2026-11-12.md)

## New exact inputs

The held test contains six generated inserts and 20 genuine controls from eight groups. The generated recordings include two DuplexGen PLN clone, two ElevenLabs v4 expressive, and two Kokoro stock voices. Every generated clip is matched to its unedited host and to two genuine splices. Each matched group uses identical PCM outside the known edit. Two isolated room recordings supply additional controls. Four expressive and stock voice profiles have unrelated prior exposure. The exact selected source recordings, declared foreground speakers, and room participants are disjoint from calibration. The two AMI meetings were used in other evaluations, with separate capture intervals here. [Source selection and rights](audio-communication-replication-2026-11-12.md)

Each six-second native 16 kHz track includes real competing room speech at a −5 dB whole-clip foreground-to-background ratio. Manual word annotations show two or more competing voices during 460–960 ms of each known edit, over at least half its duration. The 0.64-second edge edits start at 0.16 and 5.20 seconds. The four 0.80- or 0.96-second interior edits straddle a boundary in the disjoint Edge grid. The archive recordings, room captures and source identities stay local. [AMI corpus terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

## Replay

The anonymous ledger contains ordinary source-group ordinals, edit intervals and scalar window scores. The independent reviewer derives the four points, frozen and retrospective held outcomes, and localization geometry from those values.

```sh
python3 -B -S backend/scripts/review_audio_edge_window_transfer.py
```

With the sealed earlier inputs, licensed source archives and pinned voice assets available locally, use new run and held directories to reconstruct the native evaluation. The CPU model loaded once for 116 new calibration forwards and 176 held forwards. The native reviewer verifies all 66 recording bindings and 462 planned windows, checks reused scores, and confirms identical samples outside each matched edit.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_edge_window_transfer sources --held "$held"
backend/.venv/bin/python -B -m scripts.audio_edge_window_transfer prepare --run "$run" --held "$held"
backend/.venv/bin/python -B -m scripts.audio_edge_window_transfer score --run "$run" --held "$held"
backend/.venv/bin/python -B -m scripts.review_audio_edge_window_transfer --native "$run" "$held" --receipt "$run/replay.json"
```
