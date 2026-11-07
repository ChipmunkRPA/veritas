# Real passage calibration: useful probability gains, unresolved detection tradeoff

The first real-data specialist pilot completed all four frozen arms. Separate
short/long calibration plus two detectors reduced source-balanced file Brier loss
from0.146067 to0.122169 on the reserved prompt groups. It also exchanged fewer
human false alarms for more missed documents containing AI passages. This is not
enough evidence to replace the current customer check or admit these scores.

## Reserved-group results

Each arm assessed the same128 constructed documents from32 reserved prompt
components:32 human/human,32 AI/AI and64 mixed. There were no missing scores or
exact diagnostic ties. These are internally reserved groups from a development
corpus, not independently verified writers or proof of unseen model training.

| Frozen arm | File Brier, source/class balanced | False AI calls /32 human documents | Missed AI /96 documents with AI | Unassessed or tied /128 |
| --- | ---: | ---: | ---: | ---: |
| Existing detector, pooled calibration | 0.146067 | 11 | 4 | 0 |
| Existing detector, short/long calibration | 0.137012 | 11 | 7 | 0 |
| Two detectors, pooled calibration | 0.132496 | 7 | 12 | 0 |
| Two detectors, short/long calibration | 0.122169 | 6 | 15 | 0 |

The call counts use a prospectively specified internal0.5 tendency readout under
a50:50 calibration reference prior. This is not a public screening threshold,
validated upload authenticity probability or claimed customer error rate. All
four arms use the same source-group-balanced calibration recipe and separate
combination calibration. Fewer false alarms alone is not superior detection;
neither is a lower Brier loss sufficient when missed AI increases. A comparable
false-alarm operating comparison needs its own frozen procedure and new sources.

## What was actually run and checked

The private source CSV contains66,000 records. The fixed eligibility and connected
prompt/duplicate-answer audit found759 eligible components;128 were selected
before scores. Each of the four roles contains32 groups and both local classes
in both passage-length contexts, exceeding the25-group/class support floor.
Material from16 AI generators appears in the selection. No mixed-process source
row was treated as known character-level truth.

The two fixed Apache2 checkpoints each scored512 unique original passages:
384 for the three fitting/calibration roles, then128 reserved passages after all
four fitted artifacts were frozen. The four document compositions reuse those
passages within their source group. That gives512 constructed documents, not512
independent people. Each observation binds its exact copied source interval and
Unicode-codepoint position in the constructed document.

A separate standard-library reader returned to the actual CSV and verified all
512 original slices, source labels and hashes. Wrong row, wrong span and flipped
local truth were refused. All four neural runs completed without a resource guard
or journal failure. There were32 direct unbatched forward audits; the largest
absolute native-response difference was6.258487701416016e-7, below the unchanged
1e-5 native limit. Complete token coverage was checked for every scored passage.
The largest sampled process-tree memory was2,401,452,032 bytes; no cloud model
or customer upload was involved.

The model-free local scalar replay refitted every arm and reproduced2,048
confirmation file/local probabilities exactly. It also refused cross-role source
lineage, kept confirmation scores out of fitting and withheld the combined score
when a required check was skipped. The separate Linux workflow reproduces these
numerical steps only; it does not independently run the text models or examine
authorship. The [actual Linux run](https://github.com/ChipmunkRPA/veritas/actions/runs/42401818195)
passed:2,048 probabilities compared, maximum absolute difference
3.3306690738754696e-16 within the unchanged1e-6 limit, with all three adverse
invariants preserved. The [decoded job receipt](evaluation-panels/writing-specialists-v5-linux-review.json)
records the observed outcome; it is not an outside human review.

## Limits and next work across all four media

TXD-22 version1 is attributed to Md Sadiq Iqbal and Mohammod Abul Kashem,
[Mendeley Data, doi:10.17632/prcjcggtjf.1](https://data.mendeley.com/datasets/prcjcggtjf/1),
with publisher-declared CC BY4.0. Raw text stays private. The CSV's non-ASCII
encoding is inconsistent; this pilot used only unchanged ASCII source strings.
Publisher labels and consent claims were not independently verified. Writer IDs
are absent and external detector training overlap remains unknown. Artificial
passage concatenation and short English Q&A do not validate full articles,
emails, scanned documents, other languages or arbitrary paragraph boundaries.

The [protocol](writing-specialists-v5-protocol.md),
[aggregate results](evaluation-panels/writing-specialists-v5-result.json),
[original-source audit](evaluation-panels/writing-specialists-v5-original-source-audit.json),
[commercial-source review](evaluation-panels/writing-specialists-v5-commercial-source-review.json)
and compact hash/position/label/scalar packets support reproducible numerical
inspection without publishing raw text or weights. No outside human reviewer has
participated. Customer confidence, payment-before-analysis, reports, model names
and rollback remain unchanged. GCP remains last.

Images still need source-bound region/format calibrations; video needs separate
face/mouth interval calibration with source/donor isolation; voice needs real
measured-noise routing and role-safe background mixtures. Their real-data fitting
and fresh confirmation are not completed by this text pilot. All four media
remain in scope under the shared specialist architecture.
