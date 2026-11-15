# Can input sensitivity locate short voice edits under competing speech?

The existing speech check can respond to a real part of a clip and highlight
the wrong interval. This experiment tests whether the input most responsible
for a strong response helps select a more useful interval. Its signed
upper-frequency sensitivity added zero useful locations to the current review
on 40 held edits. Requiring agreement with the normal response helped on one
generator panel and lost more locations on the stock-voice panel. The
experimental methods stay outside the app.

[Phoneme-level attribution research](https://arxiv.org/abs/2607.08586) links
classifier saliency to sounds and pauses. [Speech restoration research](https://arxiv.org/abs/2603.14033)
shows why changes in a classifier response can reflect processing or voice
quality. This local experiment adopts the question of time-specific input
contribution. It uses the project's already licensed voice and speech-activity
weights and local calculations. [Asset audit](commercial-model-audit.md)

## Fixed local comparison

Each six-second mono input keeps its original mixture and speech activity gate.
The unchanged estimator reads four-second contexts at zero and two seconds.
For each 20 ms feature position, measure the local derivative of the full
context response when the upper 40 of 80 normalized Mel bands move toward
or away from the context average. The signed measurement estimates the
direction and size of a small response change. It gives no source-authenticity
probability and can respond to surrounding sound.

Compare the ordinary median frame response, the median signed contribution,
and a restricted response that permits a window only when its contribution
is at least the within-recording median. All arms use one-second windows at
half-second steps and exact 16 kHz source-sample positions. If two contexts
measure the same window, take the lower reading for each stream. Calibrate
separate operating points on 44 development recordings before scoring the
151 comparison recordings. Allow a useful location when it covers at least
half the edit and reaches 0.30 temporal intersection over union. Ordinary
flag counts are repeated deliveries, so the ledger also records source groups.
[Recorded source identities and controls](audio-shared-mel-context-2026-11-14.md)

## Held results

Each table cell gives useful edit locations and ordinary flags. The supplement
arms retain previously flagged ranges.

| Method | Random stock voices, 16 edits and 24 ordinary | Three generators, 12 edits and 36 ordinary | SINE, 12 edits and 24 ordinary |
| --- | ---: | ---: | ---: |
| Current short review | 1 / 3 | 0 / 5 | 0 / 0 |
| Plus dense response | 4 / 6 | 2 / 14 | 0 / 4 |
| Plus signed contribution | 1 / 4 | 0 / 7 | 0 / 1 |
| Plus agreement of response and contribution | 1 / 5 | 3 / 14 | 0 / 4 |

The late Chatterbox replacement ran from **5.200–5.840 s**. The agreement
arm moved the selected window to **5.000–6.000 s**, where the ordinary dense
route had selected a misplaced window. On the stock-voice panel this same
agreement rule lost three useful ranges found by dense scoring, while ordinary
flags fell by one. On SINE, one existing broad alert already covered a known
**1.510–1.850 s** splice. In an exploratory readout, either dense route could
narrow that existing alert to a useful location. Replacing broad evidence
with a nested range needs a new test that checks gains and losses on a
separate roster.

The signed measurement alone ranked a useful window first on five of 40
held edits before applying its point. Agreement ranked 11, versus 14 for the
ordinary dense response. No arm flagged an isolated held background. Three
of eight SINE background recordings had insufficient qualifying speech for
the new methods. All 40 edited inputs were assessed. The 12 held same-word
resyntheses are counted separately from the ordinary files. The standalone
contribution flagged one, agreement flagged three, and dense scoring flagged
four. All panels had prior project exposure. These results characterize the
tested mixture and generator groups.

## Reproduce

The [score ledger](evaluation-panels/audio-input-contribution-transfer-2026-11-14.json)
retains only timing, scalar values, outcome counts and hashes. Standard-library
Python rechecks the roster, exact input commitments, agreement filtering,
development points and held outcomes. A separate readback against the shared
Mel ledger matched all 2,018 measured dense windows and their availability.
The native run also checked the first available four-second context with a
symmetric one-percent perturbation. The predicted and observed derivatives
were 3.82793 and 3.82959, respectively. The mean frame response reproduced the
full model response within 0.00000024.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.audio_input_contribution_transfer review \
  docs/evaluation-panels/audio-input-contribution-transfer-2026-11-14.json
```

Use the [same local archive layouts](audio-shared-mel-context-2026-11-14.md#replay)
and pinned weights for native scoring. The original native run used
`0ed481bc`. The portable reviewer reads the published parent receipt for
the same comparison records. Run scoring in a new output directory.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_input_contribution_transfer score "$run" \
  --archive "$stock_archive" --sine "$sine_panel"
```
