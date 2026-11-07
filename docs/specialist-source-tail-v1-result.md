# A regional image arm advances; passage alternatives still lose detection

The frozen genuine-source operating comparison completed all eight image and
passage arms. The pooled regional image arm found four more edited inputs with
the same observed false alarms as its research reference. It passes all nine
development continuation gates. Passage alternatives still miss more AI content,
including a severe short-AI-first mixture failure. No customer score or serving
model changes from this study.

## Image results

The reserved development role has 60 source pairs, each genuine/edit pair shown
as PNG and JPEG quality75: 120 genuine and 120 edited inputs. All were consumed
before this new analysis. Points were chosen separately for each arm using only
60 genuine combination-calibration source groups and the same frozen procedure.
There are no missing test scores. Whole-map arms have many exact boundary ties,
which are retained and unflagged under the frozen strict comparison; they are
not original-file verdicts. The regional arms have no exact boundary ties.

| Arm | Edited inputs found /120 | Missed /120 | False AI /120 genuine | False genuine groups /60 | Boundary ties /240 | Source/class-balanced Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Whole map, pooled reference | 22 | 98 | 10 | 9 | 208 | 0.254825 |
| Whole map, format-specific | 19 | 101 | 9 | 9 | 212 | 0.253488 |
| Whole map + regions, pooled | 26 | 94 | 10 | 9 | 0 | 0.241323 |
| Whole map + regions, format-specific | 24 | 96 | 9 | 9 | 0 | 0.240569 |

The pooled regional gain is in PNG: 24/60 edits found versus20/60, with9/60 false
genuine flags in both. JPEG remains weak: both find only2/60 edits and falsely
flag1/60 genuine JPEGs. The format-specific regional arm finds24/60 PNG edits but
misses all60 JPEG edits, failing the no-condition-detection-loss gate. Its lower
overall probability loss is insufficient to hide that failure.

The same calibration target is not a promised false-alarm rate: the pooled arms
actually falsely flag9/60 genuine source groups, or15%, in this development role.
Even the passing arm misses94/120 edited inputs. Passing a relative improvement
gate is not evidence of reliable general detection. This is an experimental
TIDAL readout, including JPEG sensitivity testing, rather than the complete
serving PNG/JPEG pipeline. Its many flat, tied scores also make the whole-map
reference a weak comparator. The pooled regional arm advances first to direct
comparison with the current service's evidence check, then fresh source testing.
The JPEG gap remains required work.

## Passage results

The reserved development role has32 source/prompt components and128 constructed
documents:32 genuine-only and96 containing AI across three mixture patterns.
The operating procedure uses32 genuine combination-calibration groups per arm.
There are no unassessed or exactly tied test scores in these arms.

| Arm | AI documents found /96 | Missed /96 | False AI /32 genuine | Source/class-balanced Brier |
| --- | ---: | ---: | ---: | ---: |
| Baseline, pooled reference | 86 | 10 | 6 | 0.146067 |
| Baseline, length-specific | 60 | 36 | 0 | 0.137012 |
| Combined, pooled | 71 | 25 | 5 | 0.132496 |
| Combined, length-specific | 63 | 33 | 3 | 0.122169 |

The length-specific baseline finds30/32 AI-only documents and30/32 human-first,
AI-second mixtures, but0/32 AI-first, human-second mixtures. The combined
length-specific arm finds only7/32 of that last pattern. The baseline pooled
reference finds26/32. These constructions place60 words first and100 second;
they expose weakness when the AI contribution is the shorter passage, not proof
of a universal position effect. No alternative passes the frozen continuation
gates. Fewer false alarms and lower Brier loss do not compensate for these misses.
No candidate replaces the writing baseline.

## What was frozen and reproduced

The [protocol](specialist-source-tail-v1-protocol.md), code, runtime and original
public scalar packets were committed in
[d565cec](https://github.com/ChipmunkRPA/veritas/commit/d565ceca1355274d0e3ddd678f6b651be0a6cce4)
before this comparison's fitting or reserved-role evaluation. All eight fitted
artifacts and genuine-source points were written and hash-frozen before this
analysis's reserved-role evaluation. The rule uses one genuine calibration
source maximum and a strict upper-tail comparison; it never chooses a point
from reserved results or an ROC curve. Original diagnostic0.5 results remain
unchanged in their earlier reports. No public screening threshold is introduced.

All5,888 original file/local probabilities and their source positions, contexts
and assessment bindings replayed exactly in the local run. The separate local
numerical replay refitted all eight arms with maximum numeric difference0 and
identical counts/decisions. In every arm, changing reserved raw responses left
fitted parameters and operating points unchanged. Eleven new local regressions
cover source-tail support, duplicate/missing observations, condition failures,
prediction parity and the corrected video runner command.

The [public package](evaluation-panels/specialist-source-tail-v1-release.json)
contains aggregate results, fitted parameters, points, source commitments and
replay instructions. It reuses earlier public scalar packets and includes no raw
media, text or model weights. The [local replay receipt](evaluation-panels/specialist-source-tail-v1-local-numerical-reproduction.json)
records the actual local run. The [actual corrected Linux run](https://github.com/ChipmunkRPA/veritas/actions/runs/42413673942)
passed: all eight arms retained exactly the same counts and decisions, with
maximum numeric difference3.3306690738754696e-16, below the unchanged1e-6 limit.
Reserved-response parameter/point invariance passed for all eight arms. The
[decoded job receipt](evaluation-panels/specialist-source-tail-v1-linux-review.json)
records the completed job's actual output. Neither numerical replay is outside
human review or fresh detection validation.

The [initial Linux workflow](https://github.com/ChipmunkRPA/veritas/actions/runs/42413110055)
failed before creating a job: an unquoted colon in the one-line dependency setup
command made the workflow invalid YAML. No Linux computation or passing result
is claimed for that run. The corrected CI wrapper preserves the original
byte-frozen analysis snapshot at
[fafafe8](https://github.com/ChipmunkRPA/veritas/commit/fafafe88ab8db2bdd72d4612227bb396ab09f8ad).
It archives exactly55 public source/packet dependencies from that commit and
executes the original numerical replay there, including its originally committed
workflow bytes. The analysis implementation, parameters and original results are
unchanged. Eight additional regressions cover archive source selection, path
containment and parsing the corrected workflow. The actual local archived replay
also passed with maximum numeric difference0; its [actual receipt](evaluation-panels/specialist-source-tail-v1-local-frozen-archive-reproduction.json)
records the same55 selected source files and immutable release commitment.

From a complete repository clone with the pinned numerical dependencies, replay
the immutable study through the corrected external wrapper:

```sh
PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 python -m scripts.replay_frozen_source_tail_v1 /tmp/source-tail-review.json
```

Use a new output path for each attempt. This wrapper, rather than a direct refit
against an updated working tree, preserves the original study's byte commitments.
