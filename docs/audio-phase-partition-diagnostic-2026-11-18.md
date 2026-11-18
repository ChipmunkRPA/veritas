# Short speech edits under competing voices

At the stricter development point, adding a time-aligned enhancement view
found four of 78 known edits in the strongest competing speech, compared
with two on the original mixture. On the less severe mixtures it found
nine instead of six and flagged three additional ordinary recordings. The
comparison uses previously scored Opus sources and informs a fresh holdout.

## Method

The test compares the original mono mixture, a noise-reduced view, and the
same view when its response also exceeds the recorded mixture and any
speech-admitted residual. Enhancement runs in four-second contexts spaced
two seconds apart, with a final context aligned to the end. The estimate is
projected onto the original short-time Fourier coefficients with a gain
bounded between zero and one. Subtraction gives a time-aligned residual.
The two views sum to the recorded waveform within floating-point tolerance.
The projection can preserve edit timing without establishing who spoke the
retained words. [Component-level detector evidence](https://arxiv.org/abs/2610.17617) ·
[Pinned enhancement source and license](https://github.com/Xiaobin-Rong/gtcrn/blob/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73/LICENSE)

Every arm uses 750 ms windows spaced 500 ms apart. Windows require at least
half a window of speech on the recorded mixture and on each processed view
it scores. A displayed range needs two adjacent readings above the same
operating point. A prediction covers at least half of the known edit and has
at least 0.30 interval intersection over union to count as a match. Each
prediction and edit can participate once.

The 108 deliveries come from a previously evaluated Opus panel. At each
noise level, 13 ordinary deliveries from nine foreground groups set the
operating points for each arm. The stricter point uses the largest group
response and the wider point uses the second largest. Both use the maximum
of their assigned isolated backgrounds as a floor. Other foreground groups
form the diagnostic set. The same two recorded rooms recur across both
roles, so the isolated background deliveries do not represent that many
independent acoustic environments. [Opus source panel](audio-two-second-opus-transfer-2026-11-18.md) ·
[Background recordings and terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

## Comparison

Each condition contains 12 edited deliveries with 78 labeled intervals
and five ordinary deliveries, two of which were reprocessed. Ordinary and
isolated background alerts are counted by delivery. A column entry `a → b`
compares the original-mixture arm with the union of the mixture and the
retained view. The original-mixture arm is the matching 750 ms control.

| Competing speech | Point | Located edits / 78 | Ordinary alerts / 5 | Isolated background alerts | Edited deliveries with alerts only outside the known edits |
| --- | --- | ---: | ---: | ---: | ---: |
| −5 dB | Stricter | 6 → 9 | 0 → 3 | 1 → 1 / 15 | 1 → 2 |
| −5 dB | Wider | 6 → 11 | 0 → 4 | 1 → 1 / 15 | 1 → 3 |
| −10 dB | Stricter | 2 → 4 | 0 → 0 | 1 → 1 / 18 | 1 → 2 |
| −10 dB | Wider | 2 → 6 | 0 → 0 | 1 → 1 / 18 | 3 → 2 |

Requiring the retained response to exceed both eligible alternative views
found the same two added severe-noise locations at the stricter point. At
−5 dB that rule still added three ordinary alerts. At −10 dB the stronger
union left 74 of 78 labeled intervals without a match. These results are
separate from the current application and its confidence scoring.

## Independent recount

The public package contains scalar window readings, sample counts, anonymous
group identifiers and edit ranges. It includes no raw audio or speaker
identifiers. The recount validates each scheduled window, speech admission,
foreground group split, operating point and reported outcome. It needs
Python's standard library and the project checkout.

```bash
PYTHONPATH=backend python3 -B -S backend/scripts/review_audio_phase_partition_diagnostic.py \
  docs/evaluation-panels/audio-phase-partition-diagnostic-2026-11-18.json.gz
```

[Scalar readings and counts](evaluation-panels/audio-phase-partition-diagnostic-2026-11-18.json.gz)

The next source-separated comparison has 24 locked prompts from eight
readers, two generation families, originals, matched ordinary reprocessing,
real traffic and separately recorded competing speech. The diagnostic above
uses different inputs. The fresh comparison uses predeclared operating points
and retains its source and timing checks before scoring. [Fresh speech and word alignment source](https://zenodo.org/records/19522979)
