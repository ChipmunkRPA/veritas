# Locked mouth-edit comparison on 176 matched clips

The audio-aware checks transferred some useful localization to the GRID pilot,
but still missed most visible neural edits. The stronger of the two localized
10 of 64 edits and falsely flagged 8 of 112 non-neural controls at the frozen
primary setting. The visual-only check localized none and raised no false
flags. These are finite pilot results. The app defaults and overall confidence
calculation remain unchanged.

| Frozen research check | Edits flagged and localized (64) | Assessed edits missed (64) | False AI flags (112 controls) | Unassessed (176) |
| --- | ---: | ---: | ---: | ---: |
| Visual only | 0 | 64 | 0 | 0 |
| Visual and audio features joined | 3 | 61 | 1 | 0 |
| Visual and audio interaction | 10 | 54 | 8 | 0 |

Localization here means temporal intersection over union of at least 0.20.
It is an internal research criterion. Each detected edit met it, but that
does not mean each returned range matched the precise edit boundaries. The
three joined-feature localizations came from one rendering branch; the ten
interaction localizations split five per branch. The interaction check's eight
false flags included one untouched original, five brightness or resampling
controls, and two genuine frame retimes. Its other two frozen settings
localized 1 and 9 edits with 0 and 8 false flags, respectively. The receipt
retains all three settings and the paired changes.

All sixteen audio-delay pairs had byte-identical visual-only score and window
responses. The audio-only delay added no primary AI flag in any arm and removed
one in the interaction arm. The shared-input timing diagnostic assessed 166 of
176 cases, flagged none, and left 10 unassessed. It also produced no flags on
the deliberate delay controls. This timing measurement and the full serving
pipeline are separate comparisons.

The sixteen guarded stages scored all 176 cases and 12,320 native centres. The
first original stage was retained; the fifteen continuation stages ran the same
frozen arithmetic. No stage in the completed collection had a guard violation
or sampling error. No model or calibration was fitted. The clips came from eight
known source components and included 64 observable local neural redraws, 16
untouched originals and 96 ordinary controls. The sixteen reserve clips were
left unscored. These short, lossless redraws do not establish identity-swap
detection, achieved phonetic alignment, population error rates or a serving
improvement.

[Actual aggregate](evaluation-panels/grid-native-mouth-scores-complete-v155.json) ·
[Scoring method](grid-native-mouth-scores-v151-method.md) ·
[Unchanged continuation](grid-native-mouth-scores-v155-method.md)
