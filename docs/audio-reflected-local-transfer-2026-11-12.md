# Short speech windows with reflected context

A 1.2-second voice check found three additional known edits in gain-only audio and none under strong rain or train noise. It added no file alerts on 36 genuine mixtures or eight isolated backgrounds. The gain-only improvement came from three different synthesis families. The check remains experimental. [Saved scores and exact window commitments](evaluation-panels/audio-reflected-local-transfer-2026-11-12.json)

| Known edited files with a useful location | Current focused check | With the shorter check |
| --- | ---: | ---: |
| Gain-only | 13/18 | 16/18 |
| Rain at −5 dB | 3/18 | 3/18 |
| Train at −5 dB | 0/18 | 0/18 |

The candidate made six gain-only calls. Three found previously missed locations, one overlapped an already detected edit, and two were misplaced. It marked 3.05 additional seconds outside known edits. The existing activity check admitted at least one new focus on 15 of 18 train-edited files, but none reached the frozen point. Isolated backgrounds were never assessed. The focused check retained its previous ranges throughout.

## Method and replay

For each file, test fixed 19,200-sample native windows with a 9,600-sample hop. Each must lie completely within an inconclusive or unassessed three-second window. The existing full-recording speech activity measurement must admit at least half of the native focus. Reflect 14,400 samples on each side to reach the installed voice detector's usual input length. Reflection changes the inference input. Reported locations refer only to the unreflected 1.2 seconds. Use the highest admitted focus score when it strictly exceeds the maximum from 25 disjoint genuine or isolated selection files. The selection maximum used 71 admitted focuses. [Related reflection-based training research](https://arxiv.org/abs/2604.02913) · [Previous panel and source terms](audio-subband-partial-transfer-2026-11-12.md)

The exact 98 held files had prior project exposure. The new candidate used separate selection groups and did not use held edit positions when picking a focus. The candidate evaluated 726 possible native windows and admitted 493 for inference. Native replay independently checked every native slice and reflected input against its stored hash. The saved responses also support a standard-library replay of the operating point and outcomes.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.audio_reflected_local_transfer review \
  docs/evaluation-panels/audio-reflected-local-transfer-2026-11-12.json
```

To reproduce inference, prepare `selection` and `held` from the linked prior panels. Use the pinned local voice assets and a new directory for `run`.

```sh
export PYTHONPATH=backend VERITAS_AUDIO_DETECTOR_MODE=candidate VERITAS_BILLING_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_reflected_local_transfer run "$run" \
  --selection "$selection" --held "$held"
backend/.venv/bin/python -B -m scripts.audio_reflected_local_transfer review "$run/receipt.json" \
  --selection "$selection" --held "$held"
```
